import numpy as np 
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

class DTreeClassifier():
    warnings.filterwarnings("ignore")
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_predict(self):
        dtree = DecisionTreeClassifier()
        dtree.fit(self.X_train, self.y_train)
        self.base_dtree = dtree
        y_pred = dtree.predict(self.X_test)
        return (f"Dtree before tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    
    def hyperparam_tuning(self):
        criterion = ["gini", "entropy"]
        max_depth = list(np.arange(1, 21, 2))
        min_samples_split = list(np.arange(2, 11, 2))
        max_leaf_nodes = list(np.arange(3, 26, 2))
        grid = dict(criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, max_leaf_nodes= max_leaf_nodes)
        grid_search = HalvingGridSearchCV(
            estimator = self.base_dtree,
            param_grid=grid,
            n_jobs= -1,
            cv = 5,
            scoring = "accuracy"
        )
        gs = grid_search.fit(self.X_train, self.y_train)
        best_params = gs.best_params_
        dtree_tuned = DecisionTreeClassifier(**best_params)
        dtree_tuned.fit(self.X_train, self.y_train)
        y_pred = dtree_tuned.predict(self.X_test)
        return (f"Dtree after tuning: {classification_report(y_pred=y_pred, y_true = self.y_test)}")
    def run_dtree_models(self):
        base_dtree = self.fit_predict()
        tuned_dtree = self.hyperparam_tuning()
        print(base_dtree)
        print(tuned_dtree)
