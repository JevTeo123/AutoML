# Defining the class LogisticRegression
import pandas as pd
from tabulate import tabulate
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV

class LogReg():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_predict(self):
        logreg = LogisticRegression()
        logreg.fit(self.X_train, self.y_train)
        self.base_logreg = logreg
        y_pred = logreg.predict(self.X_test)
        return (f"LogReg before tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    
    def hyperparam_tuning(self):
        solvers = ["newton-cg", "lbfgs", "liblinear"]
        penalty = ["l2"]
        c_values = [100, 10, 1.0, 0.1, 0.01]
        grid = dict(solver = solvers, penalty = penalty, C = c_values)
        grid_search = GridSearchCV(
            estimator = self.base_logreg,
            param_grid=grid,
            n_jobs= -1,
            cv = 5,
            scoring = "accuracy"
        )
        gs = grid_search.fit(self.X_train, self.y_train)
        best_params = gs.best_params_
        logreg_tuned = LogisticRegression(**best_params)
        logreg_tuned.fit(self.X_train, self.y_train)
        y_pred = logreg_tuned.predict(self.X_test)
        return (f"LogReg after tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    def run_logreg_models(self):
        base_logreg = self.fit_predict()
        tuned_logreg = self.hyperparam_tuning()
        print(base_logreg)
        print(tuned_logreg)

    
    