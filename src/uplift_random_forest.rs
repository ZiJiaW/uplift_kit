use crate::uplift_tree::*;
use concurrent_queue::ConcurrentQueue;
use mimalloc::MiMalloc;
use polars::prelude::*;
use std::{
    sync::mpsc::{self, Sender},
    thread,
};
use threadpool::ThreadPool;
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
            eval_func: eval_func.clone(),
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
        let data = DataSet::from_parquet(data_file, treatment_col.clone(), outcome_col.clone());
        self.treatment_col = treatment_col;
        self.outcome_col = outcome_col;
        let (tx, rx) = mpsc::channel();

        self.fit_impl(n_threads, tx, data);

        for t in rx {
            println!("tree fitted");
            self.trees.push(t);
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
        );
        let pool = ThreadPool::new(num_threads as usize);
        for _ in 0..2 {
            let sender = tx.clone();
            let task_q = task_q.clone();
            let data = data.clone();
            let pool = pool.clone();
            let tree = empty_tree.clone();
            let treatment_col = self.treatment_col.clone();
            let outcome_col = self.outcome_col.clone();
            thread::spawn(move || loop {
                match task_q.pop() {
                    Ok(tree_id) => {
                        let mut cur_tree = tree.clone();
                        println!("Start fitting tree id = {}", tree_id);
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
