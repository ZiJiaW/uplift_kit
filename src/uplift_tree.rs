use std::{
    collections::{HashMap, HashSet},
    sync::mpsc,
    sync::{Arc, Mutex},
};

use polars::frame::row::Row;
use polars::prelude::*;
use rand::seq::{IteratorRandom, SliceRandom};
use threadpool::ThreadPool;

#[derive(Debug, Clone)]
pub enum EvalFunc {
    Euclidiean,
}

impl EvalFunc {
    pub fn from(s: &String) -> EvalFunc {
        match &s[..] {
            "EC" => EvalFunc::Euclidiean,
            &_ => panic!("bad eval func"),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
enum SplitValue {
    Numeric(f64),
    Str(String),
}

impl SplitValue {
    fn extract_f(&self) -> f64 {
        match self {
            &SplitValue::Numeric(v) => v,
            _ => panic!("not f64"),
        }
    }

    fn extract_s(&self) -> &String {
        match self {
            SplitValue::Str(v) => v,
            _ => panic!("not str"),
        }
    }
}

#[derive(Clone, Debug)]
struct TreeNode {
    pub col_name: String,
    pub col_idx: i32,
    pub split_value: SplitValue,
    pub prob: f64,
}

impl TreeNode {
    fn new() -> TreeNode {
        TreeNode {
            col_name: String::new(),
            col_idx: -1,
            split_value: SplitValue::Numeric(0.),
            prob: -100.,
        }
    }
}
#[derive(Debug, Clone)]
pub struct UpliftTreeModel {
    nodes: Vec<TreeNode>,
    max_depth: i32,
    min_sample_leaf: i32,
    max_features: i32,
    eval_func: EvalFunc,
    max_bins: i32,
    treatment_col: String,
    outcome_col: String,
    feature_cols: Vec<String>,
}

#[derive(Clone)]
struct RawData {
    string_cols: Vec<Vec<String>>,
    numeric_cols: Vec<Vec<f64>>,
    treatment_col: Vec<i8>,
    outcome_col: Vec<i8>,
    idx_map: HashMap<String, i32>, // + for numeric, - for string
    x_names: Vec<String>,
}

impl RawData {
    pub fn from_parquet(data_file: String, treatment_col: String, outcome_col: String) -> RawData {
        let data = LazyFrame::scan_parquet(data_file, Default::default())
            .unwrap()
            .collect()
            .unwrap();
        let mut str_cols: Vec<String> = Vec::new();
        let mut numeric_cols: Vec<String> = Vec::new();

        let feature_cols = data
            .get_column_names_owned()
            .iter()
            .filter(|&x| *x != treatment_col && *x != outcome_col)
            .map(|x| x.to_owned())
            .collect::<Vec<String>>();

        let schema = data.schema();
        for f in &feature_cols {
            let tp = schema.get(f).unwrap();
            if *tp == DataType::Utf8 {
                str_cols.push(f.to_string());
            } else if !tp.is_numeric() {
                panic!("Only numeric and string features!")
            } else {
                numeric_cols.push(f.to_string())
            }
        }

        let mut data = data.lazy();
        for f in &numeric_cols {
            if *schema.get(f).unwrap() != DataType::Float64 {
                data = data.with_column(col(f).cast(DataType::Float64))
            }
        }
        data = data.with_column(col(&treatment_col).cast(DataType::Int32));
        data = data.with_column(col(&outcome_col).cast(DataType::Int32));
        let data = data.collect().unwrap();

        let mut raw_data = RawData {
            string_cols: Vec::new(),
            numeric_cols: Vec::new(),
            idx_map: HashMap::new(),
            treatment_col: Vec::new(),
            outcome_col: Vec::new(),
            x_names: feature_cols.clone(),
        };
        let schema = data.schema();
        for f in &numeric_cols {
            assert!(schema.get(f).unwrap().is_numeric());
            raw_data.numeric_cols.push(
                data.column(f)
                    .unwrap()
                    .iter()
                    .map(|v| v.try_extract().unwrap())
                    .collect(),
            );
            raw_data
                .idx_map
                .insert(f.clone(), raw_data.numeric_cols.len() as i32 - 1);
        }
        for f in &str_cols {
            raw_data.string_cols.push(
                data.column(f)
                    .unwrap()
                    .iter()
                    .map(|v| match v {
                        AnyValue::Utf8(s) => s.to_string(),
                        _ => panic!("bad utf8 column"),
                    })
                    .collect(),
            );
            raw_data
                .idx_map
                .insert(f.clone(), -(raw_data.string_cols.len() as i32));
        }

        raw_data.treatment_col = data
            .column(&treatment_col)
            .unwrap()
            .iter()
            .map(|v| v.try_extract().unwrap())
            .collect();

        raw_data.outcome_col = data
            .column(&outcome_col)
            .unwrap()
            .iter()
            .map(|v| v.try_extract().unwrap())
            .collect();
        raw_data
    }

    fn n_rows(&self) -> usize {
        self.treatment_col.len()
    }
}

pub struct DataSet {
    // read only field
    raw_data: Arc<RawData>,
    index: Arc<Vec<usize>>,
}

impl Clone for DataSet {
    fn clone(&self) -> Self {
        return DataSet {
            raw_data: self.raw_data.clone(),
            index: self.index.clone(),
        };
    }
}

impl DataSet {
    pub fn from_parquet(data_file: String, treatment_col: String, outcome_col: String) -> DataSet {
        let raw_data = RawData::from_parquet(data_file, treatment_col, outcome_col);
        let mut index = Vec::with_capacity(raw_data.n_rows());
        for i in 0..raw_data.n_rows() {
            index.push(i);
        }
        DataSet {
            raw_data: Arc::new(raw_data),
            index: Arc::new(index),
        }
    }

    fn propose_splits(&self, f: &String, max_bins: i32) -> Vec<SplitValue> {
        let col_idx = *self.raw_data.idx_map.get(f).unwrap();
        let rng = &mut rand::thread_rng();
        if col_idx >= 0 {
            let mut split_values: Vec<SplitValue> = Vec::with_capacity(max_bins as usize);
            let mut row_index = Vec::from_iter(self.index.iter().map(|v| *v));
            let col_values = &self.raw_data.numeric_cols[col_idx as usize];
            row_index.sort_unstable_by(|a, b| col_values[*a].partial_cmp(&col_values[*b]).unwrap());
            let split_range = row_index.len() as i32 / max_bins;
            for i in 1..max_bins {
                split_values.push(SplitValue::Numeric(
                    col_values[row_index[(i * split_range) as usize]],
                ))
            }
            split_values.dedup_by(|a, b| a.extract_f() == b.extract_f());
            split_values
        } else {
            let col_values = &self.raw_data.string_cols[(-col_idx - 1) as usize];
            let mut unique_v = HashSet::new();
            self.index.iter().for_each(|v| {
                unique_v.insert(col_values[*v].clone());
            });
            unique_v
                .iter()
                .choose_multiple(rng, (max_bins - 1) as usize)
                .iter()
                .map(|&v| SplitValue::Str(v.clone()))
                .collect()
        }
    }

    fn n_rows(&self) -> usize {
        self.index.len()
    }

    fn feature_cols(&self) -> Vec<String> {
        self.raw_data.x_names.clone()
    }

    fn summary(&self) -> Vec<i32> {
        let (mut n_t, mut n_c, mut n_pt, mut n_pc) = (0, 0, 0, 0);
        let treatment_col = &self.raw_data.treatment_col;
        let outcome_col = &self.raw_data.outcome_col;
        for &i in self.index.iter() {
            if treatment_col[i] == 0 {
                n_c += 1;
                n_pc += outcome_col[i] as i32;
            } else {
                n_t += 1;
                n_pt += outcome_col[i] as i32;
            }
        }
        if n_c == 0 || n_t == 0 {
            vec![]
        } else {
            vec![n_c, n_pc, n_t, n_pt]
        }
    }

    fn split_set(&self, f: &String, v: SplitValue) -> (DataSet, DataSet) {
        let col_idx = *self.raw_data.idx_map.get(f).unwrap();
        let mut left_idx = Vec::with_capacity(self.index.len() / 2);
        let mut right_idx = Vec::with_capacity(self.index.len() / 2);
        for &i in self.index.iter() {
            if col_idx >= 0 {
                if self.raw_data.numeric_cols[col_idx as usize][i] <= v.extract_f() {
                    left_idx.push(i)
                } else {
                    right_idx.push(i)
                }
            } else {
                if &self.raw_data.string_cols[(-col_idx - 1) as usize][i] == v.extract_s() {
                    left_idx.push(i)
                } else {
                    right_idx.push(i)
                }
            }
        }
        return (
            DataSet {
                raw_data: Arc::clone(&self.raw_data),
                index: Arc::new(left_idx),
            },
            DataSet {
                raw_data: Arc::clone(&self.raw_data),
                index: Arc::new(right_idx),
            },
        );
    }
}

struct Split {
    gain: f64,
    left: Option<DataSet>,
    right: Option<DataSet>,
    left_summary: Vec<i32>,
    right_summary: Vec<i32>,
    split_col: String,
    split_value: SplitValue,
}

impl Split {
    fn new() -> Split {
        Split {
            gain: f64::MIN,
            left: None,
            right: None,
            left_summary: Vec::new(),
            right_summary: Vec::new(),
            split_col: String::new(),
            split_value: SplitValue::Numeric(0.),
        }
    }

    fn exchange(
        &mut self,
        gain: f64,
        left: DataSet,
        right: DataSet,
        left_summary: Vec<i32>,
        right_summary: Vec<i32>,
        split_col: String,
        split_value: SplitValue,
    ) {
        if gain > self.gain {
            self.left = Some(left);
            self.right = Some(right);
            self.left_summary = left_summary;
            self.right_summary = right_summary;
            self.split_col = split_col;
            self.split_value = split_value;
            self.gain = gain;
        }
    }
}

impl UpliftTreeModel {
    pub fn new(
        max_depth: i32,
        min_sample_leaf: i32,
        max_features: i32,
        eval_func: EvalFunc,
        max_bins: i32,
    ) -> UpliftTreeModel {
        UpliftTreeModel {
            nodes: vec![],
            max_depth,
            min_sample_leaf,
            max_features,
            eval_func,
            max_bins,
            treatment_col: String::new(),
            outcome_col: String::new(),
            feature_cols: Vec::new(),
        }
    }

    fn calc_score(v: &Vec<i32>, eval: &EvalFunc) -> f64 {
        assert!(v.len() == 4);
        let p = v[1] as f64 / v[0] as f64;
        let q = v[3] as f64 / v[2] as f64;
        match eval {
            &EvalFunc::Euclidiean => (p - q).powi(2),
        }
    }

    fn calc_prob(v: &Vec<i32>) -> f64 {
        assert!(v.len() == 4);
        let p = v[1] as f64 / v[0] as f64;
        let q = v[3] as f64 / v[2] as f64;
        q - p
    }

    fn calc_norm(n_c: i32, n_t: i32, n_c_left: i32, n_t_left: i32) -> f64 {
        let p_t = n_t as f64 / (n_t + n_c) as f64;
        let p_c = 1. - p_t;
        let p_c_left = n_c_left as f64 / (n_t_left + n_c_left) as f64;
        let p_t_left = 1. - p_c_left;

        (1. - p_t.powi(2) - p_c.powi(2)) * (p_c_left - p_t_left).powi(2)
            + p_t * (1. - p_t_left.powi(2))
            + p_c * (1. - p_c_left.powi(2))
            + 0.5
    }

    pub fn fit(
        &mut self,
        data: DataSet,
        treatment_col: String,
        outcome_col: String,
        pool: ThreadPool,
    ) {
        self.treatment_col = treatment_col.clone();
        self.outcome_col = outcome_col.clone();
        self.feature_cols = data.feature_cols();
        let mut tree_nodes = vec![TreeNode::new(); (1 << self.max_depth + 1) - 1];
        self.build(0, 0, &mut tree_nodes, data.summary(), data.clone(), pool);
        self.nodes = tree_nodes;
    }

    fn build(
        &self,
        cur_idx: usize,
        depth: i32,
        nodes: &mut Vec<TreeNode>,
        cur_summary: Vec<i32>,
        data: DataSet,
        pool: ThreadPool,
    ) {
        let rng = &mut rand::thread_rng();
        let cur_score = UpliftTreeModel::calc_score(&cur_summary, &self.eval_func);
        let cur_prob = UpliftTreeModel::calc_prob(&cur_summary);
        let n_c = cur_summary[0];
        let n_t = cur_summary[2];
        let best_split = Arc::new(Mutex::new(Split::new()));

        let (tx, rx) = mpsc::channel();

        for f in self
            .feature_cols
            .choose_multiple(rng, self.max_features as usize)
        {
            let data = data.clone();
            let best_split = best_split.clone();
            let sender = tx.clone();
            let min_sample_leaf = self.min_sample_leaf;
            let max_bins = self.max_bins;
            let eval = self.eval_func.clone();
            let f = f.clone();
            // evaluate split gain parallelly
            pool.execute(move || {
                let split_values = data.propose_splits(&f, max_bins);
                for v in split_values {
                    let (left, right) = data.split_set(&f, v.clone());
                    let left_summary = left.summary();
                    let right_summary = right.summary();
                    if left_summary.len() != 4
                        || right_summary.len() != 4
                        || left_summary[0] + left_summary[2] <= min_sample_leaf
                        || right_summary[0] + right_summary[2] <= min_sample_leaf
                    {
                        continue;
                    }
                    let left_score = UpliftTreeModel::calc_score(&left_summary, &eval);
                    let right_score = UpliftTreeModel::calc_score(&right_summary, &eval);
                    let p = left.n_rows() as f64 / data.n_rows() as f64;
                    let n_c_left = left_summary[0];
                    let n_t_left = left_summary[2];
                    let gain = (left_score * p + right_score * (1. - p) - cur_score)
                        / UpliftTreeModel::calc_norm(n_c, n_t, n_c_left, n_t_left);
                    best_split.lock().unwrap().exchange(
                        gain,
                        left,
                        right,
                        left_summary,
                        right_summary,
                        f.clone(),
                        v,
                    );
                    sender.send(1).unwrap();
                }
            });
        }
        drop(tx);
        // wait for split evaluation jobs done
        for _ in rx {}
        // now only this thread will hold the lock
        let split = best_split.lock().unwrap();
        if split.gain > 0. && depth < self.max_depth {
            nodes[cur_idx].col_idx = self
                .feature_cols
                .iter()
                .position(|f| f == &split.split_col)
                .unwrap() as i32;
            nodes[cur_idx].col_name = split.split_col.clone();
            nodes[cur_idx].split_value = split.split_value.clone();
            self.build(
                2 * cur_idx + 1,
                depth + 1,
                nodes,
                split.left_summary.clone(),
                split.left.clone().unwrap(),
                pool.clone(),
            );
            self.build(
                2 * cur_idx + 2,
                depth + 1,
                nodes,
                split.right_summary.clone(),
                split.right.clone().unwrap(),
                pool.clone(),
            );
        } else {
            // cur node is a leaf
            nodes[cur_idx].prob = cur_prob;
        }
    }

    pub fn predict(&self, data_file: String) -> Result<Vec<f64>, PolarsError> {
        let data = LazyFrame::scan_parquet(data_file, Default::default())?
            .select(
                &self
                    .feature_cols
                    .iter()
                    .map(|f| col(f))
                    .collect::<Vec<Expr>>(),
            )
            .collect()?;
        assert!(data.shape().1 == self.feature_cols.len());
        let data_len = data.shape().0;
        let row = &mut Row::new(vec![AnyValue::Float64(0.); self.feature_cols.len()]);
        let mut result = Vec::with_capacity(data_len);
        for i in 0..data_len {
            data.get_row_amortized(i, row)?;
            result.push(self.classify(&row.0));
        }
        Ok(result)
    }

    pub fn predict_frame(&self, data: &DataFrame) -> Result<Vec<f64>, PolarsError> {
        assert!(data.shape().1 == self.feature_cols.len());
        let data_len = data.shape().0;
        let row = &mut Row::new(vec![AnyValue::Float64(0.); self.feature_cols.len()]);
        let mut result = Vec::with_capacity(data_len);
        for i in 0..data_len {
            data.get_row_amortized(i, row)?;
            result.push(self.classify(&row.0));
        }
        Ok(result)
    }

    pub fn feature_cols(&self) -> Vec<String> {
        self.feature_cols.clone()
    }

    fn classify(&self, x: &Vec<AnyValue>) -> f64 {
        let mut cur_idx = 0;
        let nodes = &self.nodes;
        let mut cur_node = &nodes[cur_idx];
        while cur_node.prob < -1. {
            let cur_value = &x[cur_node.col_idx as usize];
            let going_left = match &cur_node.split_value {
                SplitValue::Numeric(v) => cur_value.try_extract::<f64>().unwrap() <= *v,
                SplitValue::Str(v) => *cur_value == AnyValue::Utf8(v),
            };
            cur_node = if going_left {
                cur_idx = 2 * cur_idx + 1;
                &nodes[cur_idx]
            } else {
                cur_idx = 2 * cur_idx + 2;
                &nodes[cur_idx]
            }
        }
        cur_node.prob
    }
}
