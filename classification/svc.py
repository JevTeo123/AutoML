from sklearn import svm
import numpy as np 
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV


class svc():
    warnings.filterwarnings("ignore")
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_predict(self):
        svc = svm.SVC()
        svc.fit(self.X_train, self.y_train)
        self.base_svc = svc
        y_pred = svc.predict(self.X_test)
        return (f"Support Vector Machine before tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    
    def hyperparam_tuning(self):
        var_smoothing = np.logspace(0, -9, num = 100)
        grid = dict(var_smoothing = var_smoothing)
        grid_search = HalvingGridSearchCV(
            estimator = self.base_svc,
            param_grid=grid,
            n_jobs= -1,
            cv = 5,
            scoring = "accuracy"
        )
        gs = grid_search.fit(self.X_train, self.y_train)
        best_params = gs.best_params_
        svc_tuned = svm.SVC(**best_params)
        svc_tuned.fit(self.X_train, self.y_train)
        y_pred = svc_tuned.predict(self.X_test)
        return (f"Support Vector Machine after tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    def run_svc_models(self):
        base_svc = self.fit_predict()
        tuned_svc = self.hyperparam_tuning()
        print(base_svc)
        print(tuned_svc)