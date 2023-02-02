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
from sklearn.ensemble import VotingRegressor


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
        if isinstance(X_train, np.ndarray) or isinstance(X_test, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
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
    
        def fitVotingEnsemble(self, model_dict, model_results, num_models=5):
            """Creates an ensemble of models and returns the model and the performance report

            Args:
                model_dict (dictionary): A dictionary containing the different sklearn model names and the function names
                model_results (DataFrame): A DataFrame containing the results of running eazypredict fit methods
                num_models (int, optional): Number of models to be included in the embeddding. Defaults to 5.

            Returns:
                regressor, dataframe: Returns an ensemble sklearn classifier and the results validated on the dataset
            """
            estimators = []
            ensemble_name = ""
            model_results = model_results.iloc[:, 0]
            count = 0
            for model, acc in model_results.items():
                estimators.append((model, model_dict[model]))
                ensemble_name += f"{model} "
                count += 1
                if count == num_models:
                    break
            ensemble_reg = VotingRegressor(estimators)
            ensemble_reg.fit(self.X_train, self.y_train.values.ravel())

            y_pred = ensemble_reg.predict(self.X_test)

            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r_squared = r2_score(self.y_test, self.y_pred)

            result_dict = {}
            result_dict["RMSE"] = rmse
            result_dict["R Squared"] = r_squared
            result_dict["Models"] = ensemble_name

            result_df = pd.DataFrame(result_dict, index=[0])
            return ensemble_reg, result_df
