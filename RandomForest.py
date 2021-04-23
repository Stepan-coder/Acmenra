import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class RForestClassifier:
    def __init__(self, data_df, target, train_split,step):
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
        self.set_params(step)

    def set_params(self, step):
        max_depth_params = self.get_choosed_params([i for i in range(1, self.keys_len + 1)], step) + [None]
        min_samples_leaf = self.get_choosed_params([i for i in range(1, self.keys_len)], step)
        min_samples_split = self.get_choosed_params([i for i in range(2, self.keys_len)], step)
        n_estimators = self.get_choosed_params([i * 10 for i in range(1, self.keys_len)], step)
        self.params = {'n_estimators': [25],
                       'max_depth': max_depth_params,
                       'max_features': ['auto', 'log2', None],
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'criterion': ['gini', 'entropy']}

    def get_choosed_params(self, params, step):
        first_param = params[0]
        last_param = params[-1]
        remains_params = params[1:-1]
        choosed_params = remains_params[1::step]
        choosed_params = [first_param] + choosed_params + [last_param]
        return choosed_params

    def fit(self):
        print("Learned RandomForest")
        self.rfc = RandomForestClassifier(random_state=13)
        self.grid = GridSearchCV(self.rfc, self.params, cv=5, )
        self.grid.fit(self.X_train, self.Y_train)
        best_params_ = self.grid.best_params_
        self.rf = RandomForestClassifier(max_depth=best_params_['max_depth'],
                                          max_features=best_params_['max_features'],
                                          min_samples_leaf=best_params_['min_samples_leaf'],
                                          min_samples_split=best_params_['min_samples_split'],
                                          criterion=best_params_['criterion'],
                                          random_state=13)
        self.rf.fit(self.X_train, self.Y_train)

    def get_feature_importances(self):
        self.importance = {}
        for index in range(len(self.rf.feature_importances_)):
            self.importance[self.keys[index]] = self.rf.feature_importances_[index]
        return {k: v for k, v in sorted(self.importance.items(), key=lambda item: item[1], reverse=True)}

    def get_roc_auc_score(self):
        return roc_auc_score(self.y_test, self.rf.predict(self.x_test))

    def get_mean_squared_error(self):
        return mean_squared_error(self.y_test, self.rf.predict(self.x_test))