from uplift_kit.trees import UpliftRandomForestModel
import pandas as pd


model = UpliftRandomForestModel()
data = pd.read_parquet("../train.parquet")
x_names = list(data.columns[:-2])

model.fit(
    data,
    x_names=x_names,
    treatment_col="is_treated",
    outcome_col="outcome",
    n_threads=8,
)

test = pd.read_parquet("../test.parquet")

res = model.predict(data=test, n_threads=8)


print(res[:10])

row = list(test.iloc[5, :].values)

print(row)

print(model.predict_row(row))

model.save("model.json")

newmodel = UpliftRandomForestModel()
newmodel.load("model.json")

print(newmodel.predict_row(row))

# res = model.predict(data_file="../test.parquet", n_threads=8)

# print(res[:10])

# model.save("model.json")
