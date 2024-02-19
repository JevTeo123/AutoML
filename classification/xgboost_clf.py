# Defining the class LogisticRegression
import pandas as pd
from tabulate import tabulate
from io import StringIO
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import HalvingGridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV

class xgboost_clf():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.base_xgb_clf = None
        
    def fit_predict(self):
        xgb_clf = XGBClassifier()
        xgb_clf.fit(self.X_train, self.y_train)
        self.base_xgb_clf = xgb_clf
        y_pred = xgb_clf.predict(self.X_test)
        return f"Xgboost Classifier before tuning: {classification_report(y_pred=y_pred, y_true=self.y_test)}"
    
    def hyperparam_tuning(self):
        if self.base_xgb_clf is None:
            # Call fit_predict if base_xgb_clf is not already fitted
            self.fit_predict()

        grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1]
        }
        grid_search = HalvingGridSearchCV(
            estimator=self.base_xgb_clf,
            param_grid=grid,
            n_jobs=-1,
            cv=5,
            scoring="accuracy"
        )
        gs = grid_search.fit(self.X_train, self.y_train)
        best_params = gs.best_params_

        tuned_xgb_clf = XGBClassifier(**best_params)
        tuned_xgb_clf.fit(self.X_train, self.y_train)
        
        y_pred = tuned_xgb_clf.predict(self.X_test)
        return f"Xgboost Classifier after tuning: {classification_report(y_pred=y_pred, y_true=self.y_test)}"

    def run_xgboost_models(self):
        base_xgb_clf = self.fit_predict()
        tuned_xgb_clf = self.hyperparam_tuning()
        print(base_xgb_clf)
        print(tuned_xgb_clf)