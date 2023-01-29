use std::collections::{HashMap, HashSet};

use polars::frame::row::Row;
use polars::prelude::*;
use rand::seq::{IteratorRandom, SliceRandom};

#[derive(Debug, Clone)]
pub enum EvalFunc {
    Euclidiean,
}

#[derive(PartialEq, Clone, Debug)]
enum SplitValue {
    Numeric(f32),
    Str(String),
}

impl SplitValue {
    fn extract_f(&self) -> f32 {
        match self {
            &SplitValue::Numeric(v) => v,
            _ => panic!("not f32"),
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
    pub prob: f32,
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
    feature_sample_size: i32,
    eval_func: EvalFunc,
    max_splits: i32,
    treatment_col: String,
    outcome_col: String,
    feature_cols: Vec<String>,
}

#[derive(Clone)]
pub struct RawData {
    string_cols: Vec<Vec<String>>,
    numeric_cols: Vec<Vec<f32>>,
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

    fn summary(&self) -> Vec<i32> {
        let (mut n_t, mut n_c, mut n_pt, mut n_pc) = (0, 0, 0, 0);
        for i in 0..self.treatment_col.len() {
            if self.treatment_col[i] == 0 {
                n_c += 1;
                n_pc += self.outcome_col[i] as i32;
            } else {
                n_t += 1;
                n_pt += self.outcome_col[i] as i32;
            }
        }
        if n_c == 0 || n_t == 0 {
            vec![]
        } else {
            vec![n_c, n_pc, n_t, n_pt]
        }
    }

    fn n_rows(&self) -> usize {
        self.treatment_col.len()
    }

    fn empty() -> RawData {
        RawData {
            string_cols: Vec::new(),
            numeric_cols: Vec::new(),
            treatment_col: Vec::new(),
            outcome_col: Vec::new(),
            idx_map: HashMap::new(),
            x_names: Vec::new(),
        }
    }

    fn propose_splits(&self, f: &String, max_splits: i32) -> Vec<SplitValue> {
        let col_idx = *self.idx_map.get(f).unwrap();
        let rng = &mut rand::thread_rng();
        if col_idx >= 0 {
            let mut split_values: Vec<SplitValue> = Vec::with_capacity(max_splits as usize);
            let mut col_values = self.numeric_cols[col_idx as usize].clone();
            col_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let split_range = col_values.len() as i32 / (max_splits + 1);
            for i in 0..max_splits {
                split_values.push(SplitValue::Numeric(
                    col_values[(split_range * (i + 1)) as usize],
                ))
            }
            split_values.dedup_by(|a, b| a.extract_f() == b.extract_f());
            split_values
        } else {
            let col_values = &self.string_cols[(-col_idx - 1) as usize];
            let mut unique_v = HashSet::new();
            col_values.iter().for_each(|v| {
                unique_v.insert(v.clone());
            });
            unique_v
                .iter()
                .choose_multiple(rng, max_splits as usize)
                .iter()
                .map(|&v| SplitValue::Str(v.clone()))
                .collect()
        }
    }

    fn split_set(&self, f: &String, v: SplitValue) -> (RawData, RawData) {
        let col_idx = *self.idx_map.get(f).unwrap();
        let mut left_data = RawData::empty();
        let mut right_data = RawData::empty();
        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        left_data.x_names = self.x_names.clone();
        right_data.x_names = self.x_names.clone();
        left_data.idx_map = self.idx_map.clone();
        right_data.idx_map = self.idx_map.clone();
        for i in 0..self.treatment_col.len() {
            if col_idx >= 0 {
                if self.numeric_cols[col_idx as usize][i] <= v.extract_f() {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            } else {
                if &self.string_cols[(-col_idx - 1) as usize][i] == v.extract_s() {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
        }
        // construct numeric
        for col_idx in 0..self.numeric_cols.len() {
            let cur_col = &self.numeric_cols[col_idx];
            left_data
                .numeric_cols
                .push(left_idx.iter().map(|i| cur_col[*i]).collect());
            right_data
                .numeric_cols
                .push(right_idx.iter().map(|i| cur_col[*i]).collect());
        }
        // construct string cols
        for col_idx in 0..self.string_cols.len() {
            let cur_col = &self.string_cols[col_idx];
            left_data
                .string_cols
                .push(left_idx.iter().map(|i| cur_col[*i].clone()).collect());
            right_data
                .string_cols
                .push(right_idx.iter().map(|i| cur_col[*i].clone()).collect());
        }
        left_data.treatment_col = left_idx.iter().map(|i| self.treatment_col[*i]).collect();
        left_data.outcome_col = left_idx.iter().map(|i| self.outcome_col[*i]).collect();
        right_data.treatment_col = right_idx.iter().map(|i| self.treatment_col[*i]).collect();
        right_data.outcome_col = right_idx.iter().map(|i| self.outcome_col[*i]).collect();
        (left_data, right_data)
    }
}

impl UpliftTreeModel {
    pub fn new(
        max_depth: i32,
        min_sample_leaf: i32,
        feature_sample_size: i32,
        eval_func: EvalFunc,
        max_splits: i32,
    ) -> UpliftTreeModel {
        UpliftTreeModel {
            nodes: vec![],
            max_depth,
            min_sample_leaf,
            feature_sample_size,
            eval_func,
            max_splits,
            treatment_col: String::new(),
            outcome_col: String::new(),
            feature_cols: Vec::new(),
        }
    }

    fn calc_score(&self, v: &Vec<i32>) -> f32 {
        assert!(v.len() == 4);
        let p = v[1] as f32 / v[0] as f32;
        let q = v[3] as f32 / v[2] as f32;
        match self.eval_func {
            EvalFunc::Euclidiean => (p - q).powi(2),
        }
    }

    fn calc_prob(v: &Vec<i32>) -> f32 {
        assert!(v.len() == 4);
        let p = v[1] as f32 / v[0] as f32;
        let q = v[3] as f32 / v[2] as f32;
        q - p
    }

    fn calc_norm(n_c: i32, n_t: i32, n_c_left: i32, n_t_left: i32) -> f32 {
        let p_t = n_t as f32 / (n_t + n_c) as f32;
        let p_c = 1. - p_t;
        let p_c_left = n_c_left as f32 / (n_t_left + n_c_left) as f32;
        let p_t_left = 1. - p_c_left;

        (1. - p_t.powi(2) - p_c.powi(2)) * (p_c_left - p_t_left).powi(2)
            + p_t * (1. - p_t_left.powi(2))
            + p_c * (1. - p_c_left.powi(2))
            + 0.5
    }

    pub fn fit(&mut self, data: &RawData, treatment_col: String, outcome_col: String) {
        self.treatment_col = treatment_col.clone();
        self.outcome_col = outcome_col.clone();
        self.feature_cols = data.x_names.clone();
        let mut tree_nodes = vec![TreeNode::new(); (1 << self.max_depth + 1) - 1];
        self.build(0, 0, &mut tree_nodes, data.summary(), data.clone());
        self.nodes = tree_nodes;
    }

    fn build(
        &self,
        cur_idx: usize,
        depth: i32,
        nodes: &mut Vec<TreeNode>,
        cur_summary: Vec<i32>,
        data: RawData,
    ) {
        let rng = &mut rand::thread_rng();
        let cur_score = self.calc_score(&cur_summary);
        let cur_prob = UpliftTreeModel::calc_prob(&cur_summary);
        let n_c = cur_summary[0];
        let n_t = cur_summary[2];
        let mut max_gain: f32 = f32::MIN;
        let mut best_data_left = RawData::empty();
        let mut best_data_right = RawData::empty();
        let mut cached_left_summary = Vec::new();
        let mut cached_right_summary = Vec::new();
        let mut split_col = String::new();
        let mut split_value = SplitValue::Numeric(0.);

        for f in self
            .feature_cols
            .choose_multiple(rng, self.feature_sample_size as usize)
        {
            let split_values = data.propose_splits(f, self.max_splits);
            for v in split_values {
                let (data_left, data_right) = data.split_set(f, v.clone());
                let left_summary = data_left.summary();
                let right_summary = data_right.summary();
                if left_summary.len() != 4
                    || right_summary.len() != 4
                    || left_summary[0] + left_summary[2] <= self.min_sample_leaf
                    || right_summary[0] + right_summary[2] <= self.min_sample_leaf
                {
                    continue;
                }
                let left_score = self.calc_score(&left_summary);
                let right_score = self.calc_score(&right_summary);
                let p = data_left.n_rows() as f32 / data.n_rows() as f32;
                let n_c_left = left_summary[0];
                let n_t_left = left_summary[2];
                let gain = (left_score * p + right_score * (1. - p) - cur_score)
                    / UpliftTreeModel::calc_norm(n_c, n_t, n_c_left, n_t_left);
                if gain > max_gain {
                    best_data_left = data_left;
                    best_data_right = data_right;
                    cached_left_summary = left_summary;
                    cached_right_summary = right_summary;
                    max_gain = gain;
                    split_col = f.clone();
                    split_value = v;
                }
            }
        }
        if max_gain > 0. && depth < self.max_depth {
            nodes[cur_idx].col_idx = self
                .feature_cols
                .iter()
                .position(|f| *f == split_col)
                .unwrap() as i32;
            nodes[cur_idx].col_name = split_col;
            nodes[cur_idx].split_value = split_value;
            self.build(
                2 * cur_idx + 1,
                depth + 1,
                nodes,
                cached_left_summary,
                best_data_left,
            );
            self.build(
                2 * cur_idx + 2,
                depth + 1,
                nodes,
                cached_right_summary,
                best_data_right,
            );
        } else {
            // cur node is a leaf
            nodes[cur_idx].prob = cur_prob;
        }
    }

    pub fn predict(&self, data_file: String) -> Result<Vec<f32>, PolarsError> {
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

    pub fn predict_frame(&self, data: &DataFrame) -> Result<Vec<f32>, PolarsError> {
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

    fn classify(&self, x: &Vec<AnyValue>) -> f32 {
        let mut cur_idx = 0;
        let nodes = &self.nodes;
        let mut cur_node = &nodes[cur_idx];
        while cur_node.prob < -1. {
            let cur_value = &x[cur_node.col_idx as usize];
            let going_left = match &cur_node.split_value {
                SplitValue::Numeric(v) => cur_value.try_extract::<f32>().unwrap() <= *v,
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
