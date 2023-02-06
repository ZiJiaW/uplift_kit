# uplift_kit

This package currently implements the UpliftRandomForest algorithm, and may implement more machine learning algorithm designed for uplift modeling.

**Motivation**: I tried the UpliftRandomForestModel implemented in [calsalml](https://github.com/uber/causalml), and it generally outperforms other algorithms like t-learner/x-learner. But unfortunately, causalml's implementation in python is slow and memory-consuming, which makes the training procedure rather painful.

So I implement the UpliftRandomForest algorithm in Rust and provide a python API. In my test, it is 10x~20x faster than causalml' implementation and consumes much less memory.

## Usage

Make sure Rust stable version is installed on your device. If not, follow the [ installation guide](https://www.rust-lang.org/tools/install) to get stable Rust.

Currently **uplift_kit** hasn't been published on *Pypi*, you can try it in a *pipenv* environment. And we use [maturin](https://github.com/PyO3/maturin) to build python package for Rust program.

```shell
# you may install pipenv and maturin first
pip install pipenv
pip install maturin
# activate virtualenv in current dir
pipenv shell
# compile
maturin develop --release
```

Basic usage example: 

```python
import uplift_kit

model = uplift_kit.UpliftRandomForestModel(
    n_estimators=10,
    max_features=10,
    max_depth=10,
    min_sample_leaf=100,
    eval_func="KL",
    max_bins=10,
    balance=True,
    regularization=True,
    alpha=0.9,
)

model.fit(
    data_file="train.parquet",
    treatment_col="is_treated",
    outcome_col="outcome",
    n_threads=8,
)

res = model.predict(data_file="test.parquet", n_threads=8)
print(res[:10])
```

Currently we can only use parquet files for train and predict, values in `treatment_col` must be `[0,1,2...]`, where `0` represents control sample, `[1,2...k]` indicates *k* types of treatment. `outcome_col` indicates the outcome of treatments/control, must be 0/1 (binary outcome). 

The `predict` method returns the uplift value of each treatment as a `List[List[float]]`. `res[i][k]` represents the uplift value for item `i` with treatment `k`.