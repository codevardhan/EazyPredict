import sklearn
import xgboost
import lightgbm
from tqdm import tqdm
from sklearn.utils import all_estimators
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)
import os


REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "KNeighborsRegressor",
    "NuSVR",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GaussianProcessRegressor",
    "MLPRegressor",
    "XGBRegressor",
    "LGBMRegressor",
]



class EazyRegressor:
    def __init__(self, regressors="all", save_dir=False, sort_by="rmse"):
        self.regressors = regressors
        self.save_dir = save_dir
        self.sort_by = sort_by

    def __getRegressorList(self):
        if self.regressors == "all":
            self.regressors = REGRESSORS

        regressor_list = self.regressors
        self.regressors = [e for e in all_estimators() if e[0] in regressor_list]

        if "XGBRegressor" in regressor_list:
            self.regressors.append(("XGBRegressor", xgboost.XGBRegressor))

        if "LGBMRegressor" in regressor_list:
            self.regressors.append(("LGBMRegressor", lightgbm.LGBMRegressor))

    def fit(self, X_train, y_train, X_test, y_test):

        self.__getRegressorList()

        prediction_list = {}
        model_list = {}
        model_results = {}

        for name, model in tqdm(self.regressors):
            model = model()
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)

            model_list[name] = model
            prediction_list[name] = y_pred
            if self.save_dir:
                folder_path = os.path.join(self.save_dir, "regressor_model")

                os.makedirs(folder_path, exist_ok=True)
                pickle.dump(
                    model,
                    open(os.path.join(folder_path, f"{name}_model.sav"), "wb"),
                )

            results = []

            try:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            except Exception as exception:
                rmse = None
                print("Ran into an error while calculating rmse score for " + name)
                print(exception)

            try:
                r_squared = r2_score(y_test, y_pred)
            except Exception as exception:
                r_squared = None
                print("Ran into an error while calculating r_squared for " + name)
                print(exception)

            results.append(rmse)
            results.append(r_squared)

            model_results[name] = results

        if self.sort_by == "rmse":
            model_results = dict(sorted(model_results.items(), key=lambda x: x[1]))
        elif self.sort_by == "r_squared":
            model_results = dict(
                sorted(model_results.items(), key=lambda x: x[2], reverse=True)
            )
        else:
            raise Exception("Invalid evaluation metric " + str(self.sort_by))

        result_df = pd.DataFrame(model_results).transpose()
        result_df.columns = ["RMSE", "R Squared"]

        return model_list, prediction_list, result_df
