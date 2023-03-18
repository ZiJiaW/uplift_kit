# uplift_kit

This package implements the UpliftRandomForest algorithm (same in [causalml](https://github.com/uber/causalml), but faster and less memory-consuming), and may implement more machine learning algorithm designed for uplift modeling.

Reference for detail: Piotr Rzepakowski and Szymon Jaroszewicz. Decision trees for uplift modeling with single and multiple treatments. Knowl. Inf. Syst., 32(2):303–327, August 2012.

## Usage

[uplift-kit](https://pypi.org/project/uplift-kit/) has been published on *pypi*, use `pip install uplift-kit` to install.

### Basic usage example 

```python
from uplift_kit.trees import UpliftRandomForestModel
import pandas as pd

model = UpliftRandomForestModel(
    n_estimators=10,      # number of uplift trees
    max_features=10,      # maximum number of features considered in one split
    max_depth=10,         # maximum depth of one single tree
    min_sample_leaf=100,  # minumum number of samples classified to a leaf
    eval_func="ED",       # split evaluation function, support `ED, KL, CHI`
    max_bins=10,          # maximum bins considered when calculating best split
    balance=False,        # whether to use weighted average to calculate score, False mean not
    regularization=True,  # whether to add regularization term
    alpha=0.9,            # param for the regularization term
)

data = pd.read_parquet("../train.parquet")
x_names = list(data.columns[:-2])

# model will use columns of `x_names` as features
# treatment_col should contains 0,1,2,...k, where 0 indicates control sample, 1~k means treatment 1~k.
# outcome_col should only contains 0,1 as integer values, i.e. binary outcome.
model.fit(
    data,
    x_names=x_names,
    treatment_col="treats",
    outcome_col="outcome",
    n_threads=8,
)

# In prediction, model will automatically choose the feature columns (x_names) from input dataframe. 
# It returns a numpy array, where k columns per sample indicate the uplift value for treatment k.
test = pd.read_parquet("../test.parquet")
res = model.predict(data=test[x_names], n_threads=8)
print(res[:10])
```

Values in `treatment_col` must be `[0,1,2...]`, where `0` specifically represents control sample and `[1,2...k]` indicates *k* types of treatment. 

Values in `outcome_col` indicates the outcome of treatments/control, must be 0/1 (binary outcome). 

Values in `x_names` columns can be either numeric or categorical (`str` values). Model will handle both properly.

The `predict` method returns the uplift value of each treatment as a `np.array` of shape `(n_samples, k)`. `res[i][k]` represents the uplift value for item `i` of treatment `k`.

### Other usage

You can `save` a trained model and `load` it else where for prediction.

```python
model.fit(...)
model.save("saved_model.json")

new_model = UpliftRandomForestModel()
new_model.load("saved_model.json")
new_model.predict(...)
```

In basic example, the `predict` function used multi-thread for predicting a large dataset in default. However, the `predict_row` function is suitable for predicting one single sample:

```python
res = model.predict_row([1,2,"ASIA",...]) # input a list of features, consistent with `x_names`
```

`res` will be a list of `k` uplift values for `k` treatments.