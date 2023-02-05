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
