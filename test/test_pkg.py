from uplift_kit.trees import UpliftRandomForestModel
import pandas as pd


model = UpliftRandomForestModel()
data = pd.read_parquet("../train.parquet")
x_names = list(data.columns[:-2])
data.x3_informative = data.x3_informative.astype(int)

model.fit(
    data,
    x_names=x_names,
    treatment_col="is_treated",
    outcome_col="outcome",
    n_threads=8,
)


test = pd.read_parquet("../test.parquet")
test.x3_informative = test.x3_informative.astype(int)
res = model.predict(data=test[x_names].iloc[:4, :], n_threads=1)


print(res[:10])

row = list(test[x_names].iloc[5, :].values)

print(row)

print(model.predict_row(row))

model.save("model.json")

newmodel = UpliftRandomForestModel()
newmodel.load("model.json")

print(newmodel.predict_row(row))
