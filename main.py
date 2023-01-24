import sklearn
import xgboost
import lightgbm
import tqdm
from sklearn.utils import all_estimators
import numpy as np
import pandas as pd
import pickle

CLASSIFIERS = [
    "GaussianNB",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "SVC",
    "RandomForestClassifier",
    "SGDClassifier",
    "XGBClassifier",
    "LGBMClassifier",
]

# xgboost.XGBClassifier
# lightgbm.LGBMClassifier

REGRESSORS = ["XGBRegressor", "LGBMRegressor"]

# xgboost.XGBRegressor
# lightgbm.LGBMRegressor


class EasyClassifier:
    def __init__(self, classififers="all", save_dir="output"):
        self.classifiers = classififers
        self.save_dir = save_dir

    def fit(self, X_train, y_train, X_test, y_test):

        self.classifiers = __getClassifierList()

        prediction_list = {}
        model_list = {}

        for name, model in self.classifiers:
            model = model()
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)

            model_list[name] = model
            prediction_list[name] = y_pred

            if self.save_dir:
                pickle.dump(model, open(f"{self.save_dir}/{name}_model.sav", "wb"))

        return model_list, prediction_list

        def __getClassifierList(self):
            if self.classifiers == "all":
                self.classifiers = CLASSIFIERS

            classifier_list = self.classifiers
            self.classifiers = [e for e in all_estimators() if e[0] in classifier_list]

            if "XGBClassifier" in classifier_list:
                self.classifiers.append(("XGBClassifier", xgboost.XGBClassifier))

            if "LGBMClassifier" in classifier_list:
                self.classifiers.append(("LGBMClassifier", lightgbm.LGBMClassifier))
