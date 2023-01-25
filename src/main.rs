use std::any::Any;
use std::collections::hash_map::*;

use polars::datatypes::DataType;
use polars::prelude::*;
use polars::toggle_string_cache;
use polars_plan::dsl::*;
use rand::prelude::*;
use rand::seq::SliceRandom;

#[derive(Clone, Debug)]
struct TreeNode {
    col_name: String,
    split_value: f64,
    split_str: String,
    true_branch: i32,
    false_branch: i32,
}

impl TreeNode {
    fn new() -> TreeNode {
        TreeNode {
            col_name: String::new(),
            split_value: 0.,
            split_str: String::new(),
            true_branch: -1,
            false_branch: -1,
        }
    }
}

enum EvalFunc {
    Euclidiean,
}

#[derive(PartialEq)]
enum SplitValue {
    Numeric(f64),
    Str(String),
}

struct UpliftTreeModel {
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

impl UpliftTreeModel {
    pub fn new(
        max_depth: i32,
        min_sample_leaf: i32,
        feature_sample_size: i32,
        eval_func: EvalFunc,
        max_splits: i32,
    ) -> UpliftTreeModel {
        UpliftTreeModel {
            nodes: vec![TreeNode::new(); 1 << max_depth - 1],
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

    pub fn fit(
        &mut self,
        data_file: String,
        treatment_col: String,
        outcome_col: String,
    ) -> Result<(), PolarsError> {
        self.treatment_col = treatment_col.clone();
        self.outcome_col = outcome_col.clone();

        let data = LazyFrame::scan_parquet(data_file, Default::default())?.collect()?;

        let mut str_cols: Vec<String> = Vec::new();
        let mut numeric_cols: Vec<String> = Vec::new();

        self.feature_cols = data
            .get_column_names_owned()
            .iter()
            .filter(|&x| *x != treatment_col && *x != outcome_col)
            .map(|x| x.to_owned())
            .collect();

        assert!(self.feature_sample_size <= self.feature_cols.len() as i32);

        let schema = data.schema();
        for f in &self.feature_cols {
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
        for f in &str_cols {
            data = data.with_column(col(f).cast(DataType::Categorical(None)));
        }
        for f in &numeric_cols {
            data = data.with_column(col(f).cast(DataType::Float64))
        }
        data = data.with_column(col(&self.treatment_col).cast(DataType::Int32));
        data = data.with_column(col(&self.outcome_col).cast(DataType::Int32));
        let data = data.collect()?;

        self.build(data, 0)?;
        Ok(())
    }

    fn calc_split_values(&self, col_values: &Series) -> Result<Vec<SplitValue>, PolarsError> {
        let unique_values = col_values.unique()?;
        let mut split_values: Vec<SplitValue> = Vec::new();
        let rng = &mut rand::thread_rng();
        if unique_values.len() <= self.max_splits as usize {
            if unique_values.dtype().is_numeric() {
                unique_values
                    .iter()
                    .map(|v| SplitValue::Numeric(v.try_extract::<f64>().unwrap()))
                    .for_each(|v| split_values.push(v));
            } else {
                unique_values
                    .cast(&DataType::Utf8)?
                    .utf8()?
                    .into_iter()
                    .map(|v| SplitValue::Str(v.unwrap().to_string()))
                    .for_each(|v| split_values.push(v));
            }
        } else {
            if col_values.dtype().is_numeric() {
                let split_range = col_values.len() as i32 / self.max_splits;
                col_values
                    .sort(false)
                    .iter()
                    .enumerate()
                    .for_each(|(i, v)| {
                        if (i as i32 % split_range) == split_range - 1 {
                            split_values.push(SplitValue::Numeric(v.try_extract::<f64>().unwrap()));
                        }
                    });
                split_values.dedup_by(|a, b| {
                    if let SplitValue::Numeric(v1) = a {
                        if let SplitValue::Numeric(v2) = b {
                            return *v1 == *v2;
                        }
                    }
                    false
                })
            } else {
                split_values.append(
                    &mut unique_values
                        .cast(&DataType::Utf8)?
                        .utf8()?
                        .into_iter()
                        .map(|v| SplitValue::Str(v.unwrap().to_string()))
                        .choose_multiple(rng, self.max_splits as usize),
                );
            }
        }
        Ok(split_values)
    }

    fn split_set(
        &self,
        v: SplitValue,
        f: &String,
        data: DataFrame,
    ) -> Result<(DataFrame, DataFrame), PolarsError> {
        match v {
            SplitValue::Numeric(v) => Ok((
                data.clone().lazy().filter(col(f).lt_eq(v)).collect()?,
                data.lazy().filter(col(f).gt(v)).collect()?,
            )),
            SplitValue::Str(v) => Ok((
                data.clone().lazy().filter(col(f).eq(&v[..])).collect()?,
                data.lazy().filter(col(f).neq(&v[..])).collect()?,
            )),
        }
    }

    fn summary(&self, data: &DataFrame) -> Result<Vec<i32>, PolarsError> {
        let summary_data = data
            .select([&self.treatment_col, &self.outcome_col])?
            .lazy()
            .groupby([col(&self.treatment_col)])
            .agg([
                count().alias("n"),
                col(&self.outcome_col).sum().alias("n_p"),
            ])
            .sort(&self.treatment_col, Default::default())
            .collect()?
            .to_ndarray::<Int32Type>()?;
        if summary_data.dim() != (2, 3) {
            return Ok(vec![]);
        }
        Ok(vec![
            summary_data[[0, 1]], // control n_c
            summary_data[[0, 2]], // control n_pc
            summary_data[[1, 1]], // treated n_t
            summary_data[[1, 2]], // treated n_pt
        ])
    }

    fn calc_score(&self, v: &Vec<i32>) -> f64 {
        assert!(v.len() == 4);
        let p = v[1] as f64 / v[0] as f64;
        let q = v[3] as f64 / v[2] as f64;
        match self.eval_func {
            EvalFunc::Euclidiean => (p - q).powi(2),
        }
    }

    fn build(&mut self, data: DataFrame, cur_idx: i32) -> Result<(), PolarsError> {
        let rng = &mut rand::thread_rng();
        let schema = data.schema();
        let cur_summary = self.summary(&data)?;
        let cur_score = self.calc_score(&cur_summary);

        for f in self
            .feature_cols
            .choose_multiple(rng, self.feature_sample_size as usize)
        {
            let split_values = self.calc_split_values(data.column(f)?)?;
            for v in split_values {
                let (data_left, data_right) = self.split_set(v, f, data.clone())?;
                let left_summary = self.summary(&data_left)?;
                let right_summary = self.summary(&data_right)?;
                if left_summary.len() != 4
                    || right_summary.len() != 4
                    || left_summary[0] + left_summary[2] <= self.min_sample_leaf
                    || right_summary[0] + right_summary[2] <= self.min_sample_leaf
                {
                    continue;
                }
                let left_score = self.calc_score(&left_summary);
                let right_score = self.calc_score(&right_summary);
            }
        }

        Ok(())
    }
}

fn main() {
    toggle_string_cache(true);
    println!("Hello, world!");
    let lf1 = df! (
        "a" => &["foo", "bar", "ham"],
        "b" => &[1, 2, 3]
    )
    .unwrap();
    print!(
        "schema: {:?}, {:?}",
        lf1.schema(),
        lf1.column("a").unwrap().dtype()
    );
    // let lf1 = lf1
    //     .lazy()
    //     .with_column(col("a").cast(DataType::Categorical(None)))
    //     .collect()
    //     .unwrap();
    let cola = lf1.column("a").unwrap().sort(false);
    // let cola = lf1.column("a").unwrap().unique().unwrap();
    print!("a: {:?} type: {:?}", cola, cola.dtype().is_numeric());
    print!("df: {:?}", lf1);

    let res: Vec<String> = lf1
        .column("a")
        .unwrap()
        .unique()
        .unwrap()
        .cast(&DataType::Utf8)
        .unwrap()
        .utf8()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap().to_string())
        .collect();

    let lf2 = lf1
        .lazy()
        .groupby([col("a")])
        .agg([when(col("b").eq(1))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("b_sum")])
        .collect()
        .unwrap();

    print!("{:?}", lf2)
}
