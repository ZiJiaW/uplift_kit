use crate::uplift_tree::*;
use concurrent_queue::ConcurrentQueue;
use mimalloc::MiMalloc;
use polars::prelude::*;
use std::{sync::mpsc, thread};
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub struct UpliftRandomForestModel {
    n_estimators: i32,
    max_features: i32,
    max_depth: i32,
    min_sample_leaf: i32,
    eval_func: String,
    max_bins: i32,
    treatment_col: String,
    outcome_col: String,
    trees: Vec<UpliftTreeModel>,
}

impl UpliftRandomForestModel {
    pub fn new(
        n_estimators: i32,
        max_features: i32,
        max_depth: i32,
        min_sample_leaf: i32,
        eval_func: String,
        max_bins: i32,
    ) -> UpliftRandomForestModel {
        UpliftRandomForestModel {
            n_estimators,
            max_features,
            max_depth,
            min_sample_leaf,
            eval_func,
            max_bins,
            treatment_col: String::new(),
            outcome_col: String::new(),
            trees: vec![],
        }
    }

    pub fn fit(
        &mut self,
        data_file: String,
        treatment_col: String,
        outcome_col: String,
        mut n_threads: i32,
    ) {
        if n_threads < 0 {
            n_threads = num_cpus::get() as i32;
        }
        let data = RawData::from_parquet(data_file, treatment_col.clone(), outcome_col.clone());
        self.treatment_col = treatment_col;
        self.outcome_col = outcome_col;
        let task_q = Arc::new(ConcurrentQueue::unbounded());
        let (tx, rx) = mpsc::channel();

        for i in 0..self.n_estimators {
            task_q.push(i).unwrap();
        }

        for _ in 0..n_threads {
            let max_depth = self.max_depth;
            let min_sample_leaf = self.min_sample_leaf;
            let max_bins = self.max_bins;
            let max_features = self.max_features;
            let data_inner = data.clone();
            let treatment_col_inner = self.treatment_col.clone();
            let outcome_col_inner = self.outcome_col.clone();
            let task_q_inner = task_q.clone();
            let tx_inner = tx.clone();
            println!("spawning one thread...");
            thread::spawn(move || loop {
                match task_q_inner.pop() {
                    Ok(_) => {
                        let mut single_tree = UpliftTreeModel::new(
                            max_depth,
                            min_sample_leaf,
                            max_features,
                            EvalFunc::Euclidiean,
                            max_bins,
                        );
                        println!("fit one..");
                        single_tree.fit(
                            data_inner.clone(),
                            treatment_col_inner.clone(),
                            outcome_col_inner.clone(),
                        );
                        println!("fitting done one..");
                        tx_inner.send(single_tree).unwrap();
                    }
                    Err(_) => {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for t in rx {
            self.trees.push(t);
        }
    }

    pub fn predict(&self, data_file: String, mut n_threads: i32) -> Vec<f64> {
        if n_threads < 0 {
            n_threads = num_cpus::get() as i32;
        }
        let data = LazyFrame::scan_parquet(data_file, Default::default())
            .unwrap()
            .select(
                &self
                    .trees
                    .first()
                    .unwrap()
                    .feature_cols()
                    .iter()
                    .map(|f| col(f))
                    .collect::<Vec<Expr>>(),
            )
            .collect()
            .unwrap();
        let task_q = Arc::new(ConcurrentQueue::unbounded());
        let (tx, rx) = mpsc::channel();
        for i in 0..self.n_estimators {
            task_q.push(i).unwrap();
        }
        let trees = Arc::new(self.trees.clone());
        for _ in 0..n_threads {
            let trees_inner = trees.clone();
            let tx_inner = tx.clone();
            let task_q_inner = task_q.clone();
            let data_inner = data.clone();
            println!("spawning one thread...");
            thread::spawn(move || loop {
                match task_q_inner.pop() {
                    Ok(tree_idx) => {
                        tx_inner
                            .send(
                                trees_inner[tree_idx as usize]
                                    .predict_frame(&data_inner)
                                    .unwrap(),
                            )
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
                    res[i] += preds[i];
                }
            }
            count += 1;
        }
        assert!(count == self.n_estimators);

        for i in 0..res.len() {
            res[i] /= self.n_estimators as f64;
        }
        res
    }
}
