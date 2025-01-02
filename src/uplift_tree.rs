use lockfree_object_pool::{LinearObjectPool, LinearOwnedReusable};
use noisy_float::prelude::*;
use rand::seq::{index::sample, IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::mpsc,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;

use pyo3::prelude::FromPyObject;
use pyo3::types::PyList;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvalFunc {
    Euclidean,
    KL,
    Chi,
}
impl EvalFunc {
    pub fn from(s: &String) -> EvalFunc {
        match &s[..] {
            "ED" => EvalFunc::Euclidean,
            "KL" => EvalFunc::KL,
            "CHI" => EvalFunc::Chi,
            &_ => panic!("bad eval func"),
        }
    }
}

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, FromPyObject)]
pub enum SplitValue {
    Numeric(f32),
    Str(String),
}

impl SplitValue {
    fn extract_f(&self) -> f32 {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TreeNode {
    pub col_name: String,
    pub col_idx: i32,
    pub split_value: SplitValue,
    pub prob: Vec<f64>,
    pub gain: f64,
}

impl TreeNode {
    fn new() -> TreeNode {
        TreeNode {
            col_name: String::new(),
            col_idx: -1,
            split_value: SplitValue::Numeric(0.),
            prob: Vec::new(),
            gain: 0.,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpliftTreeModel {
    nodes: Vec<TreeNode>,
    max_depth: usize,
    min_sample_leaf: usize,
    max_features: usize,
    eval_func: EvalFunc,
    max_bins: usize,
    treatment_col: String,
    outcome_col: String,
    feature_cols: Vec<String>,
    balance: bool,
    regularization: bool,
    alpha: f64,
}

#[derive(Clone)]
struct RawData {
    string_cols: Vec<Vec<String>>,
    numeric_cols: Vec<Vec<f32>>,
    treatment_col: Vec<i32>,
    outcome_col: Vec<i8>,
    idx_map: HashMap<String, i32>, // + for numeric, - for string
    x_names: Vec<String>,
    n_treatments: i32,
}

impl RawData {
    pub fn from_pylist(
        data: HashMap<String, &PyList>,
        x_names: Vec<String>,
        treatment_col: String,
        outcome_col: String,
    ) -> RawData {
        let mut raw_data = RawData {
            string_cols: Vec::new(),
            numeric_cols: Vec::new(),
            idx_map: HashMap::new(),
            treatment_col: Vec::new(),
            outcome_col: Vec::new(),
            x_names: x_names.clone(),
            n_treatments: 0,
        };

        for f in &x_names {
            if let Ok(f_col) = data.get(f).unwrap().extract::<Vec<f32>>() {
                raw_data.numeric_cols.push(f_col);
                raw_data
                    .idx_map
                    .insert(f.clone(), raw_data.numeric_cols.len() as i32 - 1);
                continue;
            }
            if let Ok(f_col) = data.get(f).unwrap().extract::<Vec<String>>() {
                raw_data.string_cols.push(f_col);
                raw_data
                    .idx_map
                    .insert(f.clone(), -(raw_data.string_cols.len() as i32));
                continue;
            }
            panic!("BAD type of column: {}", f);
        }

        raw_data.treatment_col = data.get(&treatment_col).unwrap().extract().unwrap();
        raw_data.outcome_col = data.get(&outcome_col).unwrap().extract().unwrap();
        raw_data.n_treatments = *raw_data.treatment_col.iter().max().unwrap() as i32;

        raw_data
    }

    fn n_rows(&self) -> usize {
        self.treatment_col.len()
    }
}

// DataSet is read only object
// use object pool to allocate index array
#[derive(Clone)]
pub struct DataSet {
    raw_data: Arc<RawData>,
    index: Arc<LinearOwnedReusable<Vec<usize>>>,
    index_pool: Arc<LinearObjectPool<Vec<usize>>>,
}
impl DataSet {
    pub fn from_pylist(
        data: HashMap<String, &PyList>,
        x_names: Vec<String>,
        treatment_col: String,
        outcome_col: String,
    ) -> DataSet {
        let raw_data = RawData::from_pylist(data, x_names, treatment_col, outcome_col);
        DataSet::from_raw_data(raw_data)
    }

    fn from_raw_data(raw_data: RawData) -> DataSet {
        let index_pool = Arc::new(LinearObjectPool::new(
            || Vec::<usize>::with_capacity(1024),
            |v| v.clear(),
        ));
        let mut index = index_pool.pull_owned();
        index.reserve(raw_data.n_rows());
        for i in 0..raw_data.n_rows() {
            index.push(i);
        }
        DataSet {
            raw_data: Arc::new(raw_data),
            index: Arc::new(index),
            index_pool,
        }
    }

    fn propose_splits(&self, f: &String, max_bins: usize) -> Vec<SplitValue> {
        let col_idx = *self.raw_data.idx_map.get(f).unwrap();
        let rng = &mut rand::thread_rng();
        if col_idx >= 0 {
            let mut split_values: Vec<SplitValue> = Vec::with_capacity(max_bins);
            let col_values = &self.raw_data.numeric_cols[col_idx as usize];
            // sample max_bins * 500 to propose quantile split
            let sample_size = std::cmp::min(max_bins * 500, self.index.len());
            let mut sample_values: Vec<N32> = sample(rng, self.index.len(), sample_size)
                .iter()
                .map(|i| n32(col_values[self.index[i]]))
                .collect();

            sample_values.sort_unstable();

            for i in 1..max_bins {
                split_values.push(SplitValue::Numeric(
                    sample_values[sample_size * i / max_bins].into(),
                ))
            }

            split_values.dedup_by(|a, b| a.extract_f() == b.extract_f());
            split_values
        } else {
            let col_values = &self.raw_data.string_cols[(-col_idx - 1) as usize];
            let sample_size = std::cmp::min(max_bins * 500, self.index.len());
            let mut unique_v = HashSet::new();

            sample(rng, self.index.len(), sample_size)
                .iter()
                .for_each(|i| {
                    unique_v.insert(col_values[self.index[i]].clone());
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

    fn n_treatments(&self) -> usize {
        self.raw_data.n_treatments as usize
    }

    fn feature_cols(&self) -> Vec<String> {
        self.raw_data.x_names.clone()
    }

    fn summary(&self) -> Vec<i32> {
        let treatment_col = &self.raw_data.treatment_col;
        let outcome_col = &self.raw_data.outcome_col;
        let mut res = vec![0i32; 2 * (1 + self.n_treatments())];

        for &i in self.index.iter() {
            res[2 * treatment_col[i] as usize] += 1;
            res[2 * treatment_col[i] as usize + 1] += outcome_col[i] as i32;
        }
        res
    }

    fn split_set(&self, f: &String, v: SplitValue) -> (DataSet, DataSet) {
        let col_idx = *self.raw_data.idx_map.get(f).unwrap();
        let mut left_idx = self.index_pool.pull_owned();
        let mut right_idx = self.index_pool.pull_owned();
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
                raw_data: self.raw_data.clone(),
                index: Arc::new(left_idx),
                index_pool: self.index_pool.clone(),
            },
            DataSet {
                raw_data: self.raw_data.clone(),
                index: Arc::new(right_idx),
                index_pool: self.index_pool.clone(),
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

fn ed(p: f64, q: f64) -> f64 {
    2. * (p - q).powi(2)
}

fn kl(p: f64, q: f64) -> f64 {
    if p == 0. || q == 0. {
        -0.000001_f64.log2()
    } else {
        p * (p / q).log2() + (1. - p) * ((1. - p) / (1. - q)).log2()
    }
}

fn chi(p: f64, q: f64) -> f64 {
    if p == 0. || q == 0. {
        1000000.
    } else {
        (p - q).powi(2) / q + (p - q).powi(2) / (1. - q)
    }
}

fn gini(p: f64, q: f64) -> f64 {
    1. - p.powi(2) - q.powi(2)
}

fn gini_one(p: f64) -> f64 {
    1. - p.powi(2)
}


impl UpliftTreeModel {
    pub fn new(
        max_depth: usize,
        min_sample_leaf: usize,
        max_features: usize,
        eval_func: EvalFunc,
        max_bins: usize,
        balance: bool,
        regularization: bool,
        alpha: f64,
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
            balance,
            regularization,
            alpha,
        }
    }

    pub fn feature_cols(&self) -> Vec<String> {
        return self.feature_cols.clone();
    }

    fn calc_score(v: &Vec<i32>, eval: &EvalFunc, balance: bool, cur_size: usize) -> f64 {
        assert!(v.len() % 2 == 0 && v.len() >= 4);
        let k = v.len() / 2 - 1;

        let func = match eval {
            &EvalFunc::Euclidean => ed,
            &EvalFunc::KL => kl,
            &EvalFunc::Chi => chi,
        };

        let mut score = 0.;
        let p_0 = v[1] as f64 / v[0] as f64;
        if balance {
            let l = 1. / (k as f64);
            for i in 1..k + 1 {
                score += l * func(v[2 * i + 1] as f64 / v[2 * i] as f64, p_0);
            }
        } else {
            for i in 1..k + 1 {
                let l = v[2 * i] as f64 / (cur_size as i32 - v[0]) as f64;
                score += l * func(v[2 * i + 1] as f64 / v[2 * i] as f64, p_0);
            }
        }
        score
    }

    fn calc_prob(v: &Vec<i32>) -> Vec<f64> {
        let p_0 = v[1] as f64 / v[0] as f64;
        let mut prob = Vec::with_capacity(v.len() / 2);
        for i in 1..v.len() / 2 {
            prob.push(v[2 * i + 1] as f64 / v[2 * i] as f64 - p_0);
        }
        prob
    }

    fn regularization_term(
        left_summary: &Vec<i32>,
        cur_summary: &Vec<i32>,
        n_left: usize,
        n_cur: usize,
        alpha: f64,
    ) -> f64 {
        let n_c = cur_summary[0];
        let n_t = n_cur as i32 - n_c;
        let n_c_left = left_summary[0];
        let n_t_left = n_left as i32 - n_c_left;
        let pc_left = n_c_left as f64 / n_left as f64;
        let pt_left = 1. - pc_left;
        let pca = n_c_left as f64 / n_c as f64;
        let pta = n_t_left as f64 / n_t as f64;
        let mut res = 0.5;
        // global treatment imbalance penalty
        res += alpha * gini(pc_left, pt_left) * ed(pca, pta);
        // single treatment imbalance penalty
        for k in 1..left_summary.len() / 2 {
            let pti = left_summary[2 * k] as f64 / (left_summary[2 * k] + n_c_left) as f64;
            let ptia = left_summary[2 * k] as f64 / cur_summary[2 * k] as f64;
            res += (1. - alpha) * gini(pti, 1. - pti) * ed(ptia, pca);
        }
        // panalty for large
        for k in 0..left_summary.len() / 2 {
            res += (left_summary[2 * k] as f64 / n_left as f64)
                * gini_one(left_summary[2 * k] as f64 / cur_summary[2 * k] as f64);
        }
        res
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
        depth: usize,
        nodes: &mut Vec<TreeNode>,
        cur_summary: Vec<i32>,
        data: DataSet,
        pool: ThreadPool,
    ) {
        let rng = &mut rand::thread_rng();
        let cur_score =
            UpliftTreeModel::calc_score(&cur_summary, &self.eval_func, self.balance, data.n_rows());
        let cur_prob = UpliftTreeModel::calc_prob(&cur_summary);
        let best_split = Arc::new(Mutex::new(Split::new()));

        let (tx, rx) = mpsc::channel();
        let cur_summary = Arc::new(cur_summary);

        for f in self
            .feature_cols
            .choose_multiple(rng, self.max_features)
        {
            let data = data.clone();
            let best_split = best_split.clone();
            let sender = tx.clone();
            let min_sample_leaf = self.min_sample_leaf;
            let max_bins = self.max_bins;
            let eval = self.eval_func.clone();
            let f = f.clone();
            let balance = self.balance;
            let alpha = self.alpha;
            let regularization = self.regularization;
            let cur_summary = cur_summary.clone();
            // evaluate split gain parallelly
            pool.execute(move || {
                let split_values = data.propose_splits(&f, max_bins);
                for v in split_values {
                    let (left, right) = data.split_set(&f, v.clone());
                    if left.n_rows() <= min_sample_leaf || right.n_rows() <= min_sample_leaf {
                        continue;
                    }
                    let left_summary = left.summary();
                    let right_summary = right.summary();
                    let left_score =
                        UpliftTreeModel::calc_score(&left_summary, &eval, balance, left.n_rows());
                    let right_score =
                        UpliftTreeModel::calc_score(&right_summary, &eval, balance, right.n_rows());
                    let p = left.n_rows() as f64 / data.n_rows() as f64;
                    let mut gain = left_score * p + right_score * (1. - p) - cur_score;
                    if regularization {
                        gain /= UpliftTreeModel::regularization_term(
                            &left_summary,
                            &cur_summary,
                            left.n_rows(),
                            data.n_rows(),
                            alpha,
                        );
                    }
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
            nodes[cur_idx].gain = split.gain;
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

    pub fn predict_rows(&self, data: &Vec<Vec<SplitValue>>) -> Vec<Vec<f64>> {
        assert!(data[0].len() == self.feature_cols.len());
        let mut result = Vec::with_capacity(data.len());
        for row in data {
            result.push(self.predict_row(row));
        }
        result
    }

    pub fn predict_row(&self, x: &Vec<SplitValue>) -> Vec<f64> {
        let mut cur_idx = 0;
        let nodes = &self.nodes;
        let mut cur_node = &nodes[cur_idx];
        while cur_node.prob.is_empty() {
            let cur_value = &x[cur_node.col_idx as usize];
            let going_left = match &cur_node.split_value {
                SplitValue::Numeric(v) => cur_value.extract_f() <= *v,
                SplitValue::Str(v) => cur_value.extract_s() == v,
            };
            cur_node = if going_left {
                cur_idx = 2 * cur_idx + 1;
                &nodes[cur_idx]
            } else {
                cur_idx = 2 * cur_idx + 2;
                &nodes[cur_idx]
            }
        }
        cur_node.prob.clone()
    }

    pub fn get_feature_importance(&self, importance_type: String) -> Vec<f64> {
        let mut res = vec![0.; self.feature_cols.len()];
        for node in &self.nodes {
            if node.gain <= 0. {
                continue;
            }
            if importance_type == "gain" {
                res[node.col_idx as usize] += node.gain;
            } else {
                res[node.col_idx as usize] += 1.;
            }
        }
        res
    }
}
