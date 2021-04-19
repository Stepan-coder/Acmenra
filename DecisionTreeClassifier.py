import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class DTreeClassifier:
    def __init__(self, data_df, target, train_split):
        self.data_df = data_df
        self.data_len = len(self.data_df)
        self.keys = list(self.data_df.keys())
        self.keys_len = len(self.keys)
        self.target = target
        self.train_split = train_split
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(self.data_df.drop(self.target, axis=1),
                                                                                self.data_df[self.target],
                                                                                train_size=self.train_split,
                                                                                random_state=13)
        self.set_params()

    def set_params(self):
        self.params = {'max_depth': [i for i in range(1, self.keys_len + 1)] + [None],
                       'max_features': ['auto', 'log2', None],
                       'min_samples_leaf': range (1, self.keys_len),
                       'min_samples_split': range (2, self.keys_len),
                       'criterion': ['gini', 'entropy']}

    def fit(self):
        self.dtc = DecisionTreeClassifier(random_state=13)
        self.grid = GridSearchCV(self.dtc, self.params, cv=5)
        self.grid.fit(self.X_train, self.Y_train)
        best_params_ = self.grid.best_params_
        self.dt = DecisionTreeClassifier(max_depth=best_params_['max_depth'],
                                         max_features=best_params_['max_features'],
                                         min_samples_leaf=best_params_['min_samples_leaf'],
                                         min_samples_split=best_params_['min_samples_split'],
                                         criterion=best_params_['criterion'],
                                         random_state=13)
        self.dt.fit(self.X_train, self.Y_train)

    def get_feature_importances(self):
        self.importance = {}
        for index in range(len(self.dt.feature_importances_)):
            self.importance[self.keys[index]] = self.dt.feature_importances_[index]
        return {k: v for k, v in sorted(self.importance.items(), key=lambda item: item[1], reverse=True)}
