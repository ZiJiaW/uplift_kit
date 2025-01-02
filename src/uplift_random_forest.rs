use crate::uplift_tree::*;
use concurrent_queue::ConcurrentQueue;
use mimalloc::MiMalloc;
use pyo3::types::PyList;
use pyo3::Python;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{
    sync::mpsc::{self, Sender},
    thread,
};

use std::sync::Arc;
use threadpool::ThreadPool;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Deserialize)]
pub struct UpliftRandomForestModel {
    n_estimators: i32,
    max_features: usize,
    max_depth: usize,
    min_sample_leaf: usize,
    eval_func: String,
    max_bins: usize,
    treatment_col: String,
    outcome_col: String,
    balance: bool,
    regularization: bool,
    alpha: f64,
    trees: Vec<UpliftTreeModel>,
}

fn pyprint<S: AsRef<str>>(line: S) {
    let code = format!(
        "import sys; print(\"{}\"); sys.stdout.flush();",
        line.as_ref()
    );
    Python::with_gil(|py| {
        Python::run(py, &code, None, None).unwrap();
    });
}

impl UpliftRandomForestModel {
    pub fn new(
        n_estimators: i32,
        max_features: usize,
        max_depth: usize,
        min_sample_leaf: usize,
        eval_func: String,
        max_bins: usize,
        balance: bool,
        regularization: bool,
        alpha: f64,
    ) -> UpliftRandomForestModel {
        UpliftRandomForestModel {
            n_estimators,
            max_features,
            max_depth,
            min_sample_leaf,
            eval_func: eval_func.clone(),
            max_bins,
            treatment_col: String::new(),
            outcome_col: String::new(),
            balance,
            regularization,
            alpha,
            trees: vec![],
        }
    }

    pub fn feature_cols(&self) -> Vec<String> {
        self.trees.first().unwrap().feature_cols()
    }

    pub fn fit(
        &mut self,
        data: HashMap<String, &PyList>,
        x_names: Vec<String>,
        treatment_col: String,
        outcome_col: String,
        mut n_threads: i32,
    ) {
        if n_threads < 0 {
            n_threads = num_cpus::get() as i32;
        }
        self.treatment_col = treatment_col.clone();
        self.outcome_col = outcome_col.clone();
        let data = DataSet::from_pylist(data, x_names, treatment_col, outcome_col);

        let (tx, rx) = mpsc::channel();

        pyprint(format!("Start training with {} threads...", n_threads));
        self.fit_impl(n_threads, tx, data);

        for t in rx {
            self.trees.push(t);
            pyprint(format!("Tree [{}] is finished!", self.trees.len()));
        }
    }

    fn fit_impl(&self, num_threads: i32, tx: Sender<UpliftTreeModel>, data: DataSet) {
        let task_q = Arc::new(ConcurrentQueue::unbounded());
        for i in 0..self.n_estimators {
            task_q.push(i).unwrap();
        }
        let empty_tree = UpliftTreeModel::new(
            self.max_depth,
            self.min_sample_leaf,
            self.max_features,
            EvalFunc::from(&self.eval_func),
            self.max_bins,
            self.balance,
            self.regularization,
            self.alpha,
        );
        let pool = ThreadPool::new(num_threads as usize);
        let tree_parallel_num =
            2.max(1 + f64::ceil(num_threads as f64 / self.max_features as f64) as usize);
        for _ in 0..tree_parallel_num {
            let sender = tx.clone();
            let task_q = task_q.clone();
            let data = data.clone();
            let pool = pool.clone();
            let tree = empty_tree.clone();
            let treatment_col = self.treatment_col.clone();
            let outcome_col = self.outcome_col.clone();
            thread::spawn(move || loop {
                match task_q.pop() {
                    Ok(_) => {
                        let mut cur_tree = tree.clone();
                        cur_tree.fit(
                            data.clone(),
                            treatment_col.clone(),
                            outcome_col.clone(),
                            pool.clone(),
                        );
                        sender.send(cur_tree).unwrap();
                    }
                    Err(_) => {
                        break;
                    }
                }
            });
        }
    }

    pub fn predict_rows(&self, data: Vec<Vec<SplitValue>>, mut n_threads: i32) -> Vec<Vec<f64>> {
        if n_threads < 0 {
            n_threads = num_cpus::get() as i32;
        }
        if n_threads == 1 {
            return data.iter().map(|x| self.predict_row(x)).collect();
        }
        let task_q = Arc::new(ConcurrentQueue::unbounded());
        let (tx, rx) = mpsc::channel();
        for i in 0..self.n_estimators {
            task_q.push(i).unwrap();
        }
        let trees = Arc::new(self.trees.clone());
        let data = Arc::new(data);
        for _ in 0..n_threads {
            let trees_inner = trees.clone();
            let tx_inner = tx.clone();
            let task_q_inner = task_q.clone();
            let data = data.clone();
            thread::spawn(move || loop {
                match task_q_inner.pop() {
                    Ok(tree_idx) => {
                        tx_inner
                            .send(trees_inner[tree_idx as usize].predict_rows(&data))
                            .unwrap();
                    }
                    Err(_) => {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut res = Vec::new();
        let mut count = 0;
        for preds in rx {
            if res.is_empty() {
                res = preds
            } else {
                for i in 0..res.len() {
                    for j in 0..res[i].len() {
                        res[i][j] += preds[i][j];
                    }
                }
            }
            count += 1;
        }
        assert!(count == self.n_estimators);

        for i in 0..res.len() {
            for j in 0..res[i].len() {
                res[i][j] /= self.n_estimators as f64;
            }
        }
        res
    }

    pub fn predict_row(&self, x: &Vec<SplitValue>) -> Vec<f64> {
        let mut res = Vec::new();
        for tree in &self.trees {
            let preds = tree.predict_row(x);
            if res.is_empty() {
                res = preds;
            } else {
                for i in 0..res.len() {
                    res[i] += preds[i]
                }
            }
        }
        for i in 0..res.len() {
            res[i] /= self.n_estimators as f64;
        }
        return res;
    }

    pub fn get_feature_importance(&self, importance_type: String) -> Vec<f64> {
        let mut res = vec![0.0; self.feature_cols().len()];
        for tree in &self.trees {
            let imp = tree.get_feature_importance(importance_type.clone());
            for i in 0..res.len() {
                res[i] += imp[i];
            }
        }
        for i in 0..res.len() {
            res[i] /= self.n_estimators as f64;
        }
        res
    }
}
