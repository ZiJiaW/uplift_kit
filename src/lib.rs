use pyo3::{prelude::*, types::PyList};
mod uplift_random_forest;
mod uplift_tree;
use std::{
    fs::{self, File},
    io::Write,
};
use uplift_random_forest::UpliftRandomForestModel;
use uplift_tree::SplitValue;

#[pyclass(name = "_UpliftRandomForestModel")]
struct _UpliftRandomForestModel {
    inner_model: UpliftRandomForestModel,
}

#[pymethods]
impl _UpliftRandomForestModel {
    #[new]
    fn new(
        n_estimators: i32,
        max_features: i32,
        max_depth: i32,
        min_sample_leaf: i32,
        eval_func: String,
        max_bins: i32,
        balance: bool,
        regularization: bool,
        alpha: f64,
    ) -> _UpliftRandomForestModel {
        _UpliftRandomForestModel {
            inner_model: UpliftRandomForestModel::new(
                n_estimators,
                max_features,
                max_depth,
                min_sample_leaf,
                eval_func,
                max_bins,
                balance,
                regularization,
                alpha,
            ),
        }
    }

    fn load(&mut self, path: String) {
        let json_string = fs::read_to_string(path).unwrap();
        let model = serde_json::from_str(&json_string).unwrap();
        self.inner_model = model;
    }

    fn fit(
        &mut self,
        data_file: String,
        treatment_col: String,
        outcome_col: String,
        n_threads: i32,
    ) {
        self.inner_model
            .fit(data_file, treatment_col, outcome_col, n_threads);
    }

    fn predict(&self, data_file: String, n_threads: i32) -> Vec<Vec<f64>> {
        self.inner_model.predict(data_file, n_threads)
    }

    fn predict_row(&self, x: &PyList) -> Vec<f64> {
        let row: Vec<SplitValue> = x.extract().unwrap();
        self.inner_model
            .predict_row(&row.iter().map(|v| v.to_any()).collect())
    }

    fn save(&self, path: String) {
        let json_model = serde_json::to_string(&self.inner_model).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(json_model.as_bytes()).unwrap();
    }
}

#[pymodule]
fn uplift_kit(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<_UpliftRandomForestModel>()?;
    Ok(())
}
