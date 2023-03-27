from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from copy import deepcopy


class IAUMEstimator:
    def __init__(
        self,
        control_outcome_model=None,
        treatment_outcome_model=None,
        proxy_outcome_model=None,
    ) -> None:
        if control_outcome_model is None:
            control_outcome_model = XGBRegressor(objective="binary:logistic")
        if treatment_outcome_model is None:
            treatment_outcome_model = XGBRegressor(objective="binary:logistic")
        if proxy_outcome_model is None:
            proxy_outcome_model = XGBRegressor(objective="reg:squarederror")
        self.control_outcome_model = control_outcome_model
        self.treatment_outcome_model = treatment_outcome_model
        self.proxy_outcome_model = proxy_outcome_model

    def fit(self, X, treatment, y, p, cv=5, random_state=None):
        X, treatment, y, p = (
            X.to_numpy(),
            treatment.to_numpy(),
            y.to_numpy(),
            p.to_numpy(),
        )
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        splits = cv.split(X)
        X_shuffled, y_IAUM = [], []
        for train, test in splits:
            X_train, X_test = X[train], X[test]
            treatment_train, treatment_test = treatment[train], treatment[test]
            y_train, y_test = y[train], y[test]
            X_train_0 = X_train[treatment_train == 0]
            X_train_1 = X_train[treatment_train == 1]
            y_train_0 = y_train[treatment_train == 0]
            y_train_1 = y_train[treatment_train == 1]
            treatment_outcome_model = deepcopy(self.treatment_outcome_model)
            control_outcome_model = deepcopy(self.control_outcome_model)
            treatment_outcome_model.fit(X_train_1, y_train_1)
            control_outcome_model.fit(X_train_0, y_train_0)

            mu1 = treatment_outcome_model.predict(X_test)
            mu0 = control_outcome_model.predict(X_test)

            y_test_1 = y_test[treatment_test == 1]
            y_test_0 = y_test[treatment_test == 0]
            mu1_1 = mu1[treatment_test == 1]
            mu0_0 = mu0[treatment_test == 0]
            mu1_loss = log_loss(y_test_1, mu1_1), roc_auc_score(y_test_1, mu1_1)
            mu0_loss = log_loss(y_test_0, mu0_0), roc_auc_score(y_test_0, mu0_0)
            print(f"mu1_loss: {mu1_loss}, mu0_loss: {mu0_loss}")

            p_test_1 = p[test][treatment_test == 1]
            p_test_0 = p[test][treatment_test == 0]

            y_IAUM_0 = (
                mu1[treatment_test == 0] - y_test_0 + p_test_0 * (mu0_0 - y_test_0)
            )
            y_IAUM_1 = (
                y_test_1
                - mu0[treatment_test == 1]
                + (1 - p_test_1) * (y_test_1 - mu1_1)
            )
            X_shuffled += [X_test[treatment_test == 0], X_test[treatment_test == 1]]
            y_IAUM += [y_IAUM_0, y_IAUM_1]

        X_shuffled = np.concatenate(X_shuffled, axis=0)
        y_IAUM = np.concatenate(y_IAUM, axis=0)

        self.proxy_outcome_model.fit(X_shuffled, y_IAUM)
        y_pred = self.proxy_outcome_model.predict(X_shuffled)
        print(f"IAUM train MSE loss: {mean_squared_error(y_IAUM, y_pred)}")

    def predict(self, X):
        return self.proxy_outcome_model.predict(X.to_numpy())
