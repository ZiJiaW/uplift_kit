mod uplift_random_forest;
mod uplift_tree;

use std::fs::File;
use std::io::Write;

fn main() {
    let train_data_file = "train.parquet";
    let test_data_file = "test.parquet";
    let mut model =
        uplift_random_forest::UpliftRandomForestModel::new(16, 10, 10, 100, "EC".to_string(), 12);
    let tick = std::time::SystemTime::now();
    println!("start fitting!");
    model.fit(
        train_data_file.to_string(),
        "is_treated".to_string(),
        "outcome".to_string(),
        8,
    );
    let tock = std::time::SystemTime::now();
    println!("time cost: {:?}", tock.duration_since(tick).unwrap());
    let res = model.predict(test_data_file.to_string(), 8);
    println!("res: {:?}", &res[0..10]);
    let mut f = File::create("result.csv").unwrap();
    f.write(b"uplift\n").unwrap();
    f.write_all(
        res.iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join("\n")
            .as_bytes(),
    )
    .unwrap();
}
