use pyo3::prelude::*;
mod uplift_random_forest;
mod uplift_tree;

#[pyclass(name = "UpliftRandomForestModel")]
struct _UpliftRandomForestModel {
    inner_model: uplift_random_forest::UpliftRandomForestModel,
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
            inner_model: uplift_random_forest::UpliftRandomForestModel::new(
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
}

#[pymodule]
fn uplift_kit(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<_UpliftRandomForestModel>()?;
    Ok(())
}
