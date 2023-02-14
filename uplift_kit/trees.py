from __future__ import annotations
from uplift_kit.uplift_kit import _UpliftRandomForestModel
import pandas as pd
import tempfile
import numpy as np


class UpliftRandomForestModel:
    def __init__(
        self,
        n_estimators=10,
        max_features=10,
        max_depth=8,
        min_sample_leaf=100,
        eval_func="ED",
        max_bins=10,
        balance=False,
        regularization=True,
        alpha=0.9,
    ) -> None:
        self.__model = _UpliftRandomForestModel(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_sample_leaf=min_sample_leaf,
            eval_func=eval_func,
            max_bins=max_bins,
            balance=balance,
            regularization=regularization,
            alpha=alpha,
        )

    def load(self, model_path: str):
        """
        Load the model file created by `save`. Note that loaded model cannot fit again.
        """
        self.__model.load(model_path)

    def save(self, model_path: str):
        """
        Save the trained model to `model_path`.

        Example: `model.save("mdl_xx.json")`
        """
        self.__model.save(model_path)

    def predict_row(self, row: list) -> list:
        """
        Do prediction on one sample.
        Must ensure that schema of `row` be consistent with `x_names`.

        param `row`: list of features, numeric or string.

        return: uplift value for each treatment.
        """
        return np.array(self.__model.predict_row(row))

    def fit(
        self,
        data: pd.DataFrame,
        x_names: list,
        treatment_col: str,
        outcome_col: str,
        n_threads: int = -1,
    ):
        """
        Fit with training data.

        param `data`: all columns to use.

        param `x_names`: feature column names, the order should be consistent with prediction data.

        param `treatment_col`: column name indicating treatments. This column should be `int` values. You should use `0` to represent control samples. In single-treatment case, use `1` to represent treated samples. In multi-treatment case, use `1`, `2`,.. `k` to represent `k` different treatments.

        param `outcome_col`: column name for outcome. This column should only contain `0/1` integer values, because this model only support binary outcome.

        param `n_thread`: num of threads to be used.

        """
        truncated_data = data[x_names + [treatment_col, outcome_col]]
        print("Loading data...")
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = tmpdir + "/tmp_train.parquet"
            truncated_data.to_parquet(train_path)
            self.__model.fit(
                data_file=train_path,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                n_threads=n_threads,
            )

    def predict(self, data: pd.DataFrame, n_threads: int = -1) -> np.array:
        """
        Predict for a data frame. Threads will be created to speed up prediction, so this function is suitable for processing large data. For small data, use `predict_row` instead.

        param `data`: feature columns, should be consistent with `x_names`.

        param `n_threads`: num of threads to be used.

        return: uplift value for each treatment per sample.

        """
        with tempfile.TemporaryDirectory() as tmpdir:
            predict_path = tmpdir + "/tmp_predict.parquet"
            data.to_parquet(predict_path)
            result = self.__model.predict(predict_path, n_threads)
        return np.array(result)
