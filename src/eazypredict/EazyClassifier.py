import sklearn
import xgboost
import lightgbm
from tqdm import tqdm
from sklearn.utils import all_estimators
from sklearn.ensemble import VotingClassifier

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import os

CLASSIFIERS = [
    "SGDClassifier",
    "RidgeClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "SVC",
    "RandomForestClassifier",
    "MLPClassifier",
    "XGBClassifier",
    "LGBMClassifier",
]


class EazyClassifier:
    def __init__(
        self,
        classififers="all",
        save_dir=False,
        sort_by="accuracy",
        return_model=False,
        return_predictions=False,
    ):
        self.classifiers = classififers
        self.save_dir = save_dir
        self.sort_by = sort_by

    def __getClassifierList(self):
        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS

        classifier_list = self.classifiers
        self.classifiers = [e for e in all_estimators() if e[0] in classifier_list]

        if "XGBClassifier" in classifier_list:
            self.classifiers.append(("XGBClassifier", xgboost.XGBClassifier))

        if "LGBMClassifier" in classifier_list:
            self.classifiers.append(("LGBMClassifier", lightgbm.LGBMClassifier))

    def fit(self, X_train, y_train, X_test, y_test):
        if isinstance(X_train, np.ndarray) or isinstance(X_test, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
        self.__getClassifierList()

        prediction_list = {}
        model_list = {}
        model_results = {}

        for name, model in tqdm(self.classifiers):
            model = model()
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)

            model_list[name] = model
            prediction_list[name] = y_pred

            if self.save_dir:
                folder_path = os.path.join(self.save_dir, "classifier_model")

                os.makedirs(folder_path, exist_ok=True)
                pickle.dump(
                    model,
                    open(os.path.join(folder_path, f"{name}_model.sav"), "wb"),
                )

            results = []

            try:
                accuracy = accuracy_score(y_test, y_pred)
            except Exception as exception:
                accuracy = None
                print("Ran into an error while calculating accuracy for " + name)
                print(exception)
            try:
                f1 = f1_score(y_test, y_pred, average="weighted")
            except Exception as exception:
                f1 = None
                print("Ran into an error while calculating f1 score for " + name)
                print(exception)
            try:
                roc_auc = roc_auc_score(y_test, y_pred)
            except Exception as exception:
                roc_auc = None
                print("Ran into an error while calculating ROC AUC for " + name)
                print(exception)

            results.append(accuracy)
            results.append(f1)
            results.append(roc_auc)

            model_results[name] = results

        if self.sort_by == "accuracy":
            sort_key = 1
        elif self.sort_by == "f1_score":
            sort_key = 2
        elif self.sort_key == "roc_score":
            sort_key = 3
        else:
            raise Exception("Invalid evaluation metric " + str(self.sort_by))

        model_results = dict(
            sorted(model_results.items(), key=lambda x: x[sort_key], reverse=True)
        )

        result_df = pd.DataFrame(model_results).transpose()
        result_df.columns = ["Accuracy", "f1 score", "ROC AUC score"]

        return model_list, prediction_list, result_df

    def fitVotingEnsemble(self, model_dict, model_results, num_models=5):
        """Creates an ensemble of models and returns the model and the performance report

        Args:
            model_dict (dictionary): A dictionary containing the different sklearn model names and the function names
            model_results (DataFrame): A DataFrame containing the results of running eazypredict fit methods
            num_models (int, optional): Number of models to be included in the embeddding. Defaults to 5.

        Returns:
            classifier, dataframe: Returns an ensemble sklearn classifier and the results validated on the dataset
        """
        estimators = []
        ensemble_name = ""
        model_results = model_results.iloc[:, 0]
        count=0
        for model, acc in model_results.items():
            estimators.append((model, model_dict[model]))
            ensemble_name += f"{model} "
            count+=1
            if count == num_models:
                break
            
        ensemble_clf = VotingClassifier(estimators, voting="hard")
        ensemble_clf.fit(X_train, y_train.values.ravel())
        
        y_pred = ensemble_clf.predict(X_test)
            
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        result_dict = {}
        result_dict["Accuracy"] = accuracy
        result_dict["F1 score"] = f1
        result_dict["ROC AUC score"] = roc_auc
        result_dict["Name"] = ensemble_name
        
        result_df = pd.DataFrame(result_dict, index=[0])
        return ensemble_clf, result_df 
        
if __name__ == "__main__":
    clf = EazyClassifier()
    