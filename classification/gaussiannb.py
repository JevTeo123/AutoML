import numpy as np 
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

class NB():
    warnings.filterwarnings("ignore")
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_predict(self):
        gaussiannb = GaussianNB()
        gaussiannb.fit(self.X_train, self.y_train)
        self.base_gaussian = gaussiannb
        y_pred = gaussiannb.predict(self.X_test)
        return (f"Gaussian Naive Bayes before tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    
    def hyperparam_tuning(self):
        var_smoothing = np.logspace(0, -9, num = 100)
        grid = dict(var_smoothing = var_smoothing)
        grid_search = HalvingGridSearchCV(
            estimator = self.base_gaussian,
            param_grid=grid,
            n_jobs= -1,
            cv = 5,
            scoring = "accuracy"
        )
        gs = grid_search.fit(self.X_train, self.y_train)
        best_params = gs.best_params_
        gaussiannb_tuned = GaussianNB(**best_params)
        gaussiannb_tuned.fit(self.X_train, self.y_train)
        y_pred = gaussiannb_tuned.predict(self.X_test)
        return (f"Gaussian Naive Bayes after tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    def run_gaussian_models(self):
        base_gaussiannb = self.fit_predict()
        tuned_gaussiannb = self.hyperparam_tuning()
        print(base_gaussiannb)
        print(tuned_gaussiannb)
