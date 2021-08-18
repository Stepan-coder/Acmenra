import os
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class RForestClassifier:
    def __init__(self, data_df, target, train_split, show=False):
        self.importance = {}
        self.is_grif_fit = False
        self.is_model_fit = False
        self.grid_best_params = None
        self.show = show
        self.data_len = len(data_df)
        self.keys = list(data_df.keys())
        self.keys_len = len(self.keys)
        self.target = target
        self.train_split = train_split
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(data_df.drop(self.target, axis=1),
                                                                                data_df[self.target],
                                                                                train_size=self.train_split,
                                                                                random_state=13)
        self.types = {'n_estimators': int,
                      'max_depth': int,
                      'max_features': str,
                      'min_samples_leaf': int,
                      'min_samples_split': int,
                      'criterion': str}

    def __str__(self):
        table = PrettyTable()
        is_fited = self.__is_model_fit
        table.title = f"{'Untrained ' if not self.__is_model_fit else ''}\"{self.__text_name}\" model"
        table.field_names = ["Error", "Result"]
        if self.__is_model_fit:
            table.add_row(["ROC AUC score", self.get_roc_auc_score()])
            table.add_row(["R-Squared_error", self.get_r_squared_error()])
            table.add_row(["Mean Absolute Error", self.get_mean_absolute_error()])
            table.add_row(["Mean Squared Error", self.get_mean_squared_error()])
            table.add_row(["Median Absolute Error", self.get_median_absolute_error()])
        return str(table)

    def fit_grid(self, params_dict=None, n_estimators=None,  max_depth=None, max_features=None, min_samples_leaf=None,
                 min_samples_split=None, criterion=None, step=1, cross_validation=5):
        print("Learning GridSearch RandomForestClassifier...")
        is_params_none = max_depth is None and max_features is None and min_samples_leaf is None and \
                         min_samples_split is None and criterion is None
        if params_dict is not None and is_params_none:
            params = params_dict
            for param in params:
                self.check_param(param, params[param], self.types)
        elif params_dict is None and not is_params_none:
            params = {}
            if n_estimators is not None:
                self.check_param('n_estimators', n_estimators, self.types)
                params['n_estimators'] = self.get_choosed_params(n_estimators, step)
            if max_depth is not None:
                self.check_param('max_depth', max_depth, self.types)
                params['max_depth'] = self.get_choosed_params(max_depth, step)
            if max_features is not None:
                self.check_param('max_features', max_features, self.types)
                params['max_features'] = max_features
            if min_samples_leaf is not None:
                self.check_param('min_samples_leaf', min_samples_leaf, self.types)
                params['min_samples_leaf'] = self.get_choosed_params(min_samples_leaf, step)
            if min_samples_split is not None:
                self.check_param('min_samples_split', min_samples_split, self.types)
                params['min_samples_split'] = self.get_choosed_params(min_samples_split, step)
            if criterion is not None:
                self.check_param('criterion', criterion, self.types)
                params['criterion'] = criterion
        elif params_dict is None and is_params_none:
            params = {'n_estimators': self.get_choosed_params([i * 10 for i in range(1, self.keys_len + 1)], step),
                      'max_depth': self.get_choosed_params([i for i in range(1, self.keys_len + 1)], step) + [None],
                      'max_features': ['sqrt', 'auto', 'log2', None],
                      'min_samples_leaf': self.get_choosed_params([i for i in range(1, self.keys_len + 1)], step),
                      'min_samples_split': self.get_choosed_params([i for i in range(2, self.keys_len + 1)], step),
                      'criterion': ['gini', 'entropy']}
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        if self.show:
            self.show_grid_params(params)
        rfc = RandomForestClassifier(random_state=13)
        grid = GridSearchCV(rfc, params, cv=cross_validation)
        grid.fit(self.X_train, self.Y_train)
        self.grid_best_params = grid.best_params_
        self.is_grif_fit = True

    def fit(self, grid_params=False, params=None, n_estimators=None, max_depth=None, max_features=None,
                  min_samples_leaf=None, min_samples_split=None, criterion=None):
        print("Learned RandomForestClassifier...")
        is_params_none = max_depth is None and max_features is None and min_samples_leaf is None and \
                         min_samples_split is None and criterion is None
        if grid_params and is_params_none and params is None:
            self.rf = RandomForestClassifier(n_estimators=self.grid_best_params['n_estimators'],
                                             max_depth=self.grid_best_params['max_depth'],
                                             max_features=self.grid_best_params['max_features'],
                                             min_samples_leaf=self.grid_best_params['min_samples_leaf'],
                                             min_samples_split=self.grid_best_params['min_samples_split'],
                                             criterion=self.grid_best_params['criterion'],
                                             random_state=13)
        elif not grid_params and not is_params_none and params is None:
            self.rf = RandomForestClassifier(n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             max_features=max_features,
                                             min_samples_leaf=min_samples_leaf,
                                             min_samples_split=min_samples_split,
                                             criterion=criterion,
                                             random_state=13)
        elif not grid_params and is_params_none and params is not None:
            for param in params:
                if param not in self.types:
                    raise Exception('The parameter "{0}" not in params list'.format(param))
            self.rf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                             max_depth=params['max_depth'],
                                             max_features=params['max_features'],
                                             min_samples_leaf=params['min_samples_leaf'],
                                             min_samples_split=params['min_samples_split'],
                                             criterion=params['criterion'],
                                             random_state=13)
        elif not grid_params and is_params_none and params is None:
            self.rf = RandomForestClassifier()
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        self.rf.fit(self.X_train, self.Y_train)
        self.is_model_fit = True

    def get_best_params(self):
        if self.is_grif_fit:
            return self.grid_best_params
        else:
            raise Exception('At first you need to learn grid')

    def get_feature_importances(self):
        if not self.is_model_fit:
            raise Exception('You haven"t trained the RandomForestClassifier yet')
        for index in range(len(self.rf.feature_importances_)):
            self.importance[self.keys[index]] = self.rf.feature_importances_[index]
        return {k: v for k, v in sorted(self.importance.items(), key=lambda item: item[1], reverse=True)}

    def get_roc_auc_score(self) -> float:
        """
        This method calculates the "ROC AUC score" for the {self.__text_name} on the test data
        :return: ROC AUC Score
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_roc_auc_score(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"ROC AUC score\" error")
        return error

    def get_r_squared_error(self) -> float:
        """
        This method calculates the "R-Squared_error" for the on the test data
        :return: R-Squared_error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_r_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"R-Squared_error\" error")
        return error

    def get_mean_absolute_error(self) -> float:
        """
        This method calculates the "mean_absolute_error" for the {self.text_name} on the test data
        :return: Mean Absolute Error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_absolute_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"Mean Absolute Error\" error")
        return error

    def get_mean_squared_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        :return: Mean Squared Error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"Mean Squared Error\" error")
        return error

    def get_median_absolute_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        :return: Median Absolute Error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_median_absolute_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"Median Absolute Error\" error")
        return error

    def show_grid_params(self, params):
        print("Learning GridSearch RandomForestClassifier...")
        count_elements = []
        multiply = 1
        for param in params:
            print("Param({0}[{2}]): {1}".format(param, params[param], len(params[param])))
            count_elements.append(len(params[param]))
        for ce in count_elements:
            multiply *= ce
        print("Total({1}): {0}".format(" X ".join([str(e) for e in count_elements]), multiply))

    def get_choosed_params(self, params, step):
        first_param = params[0]
        last_param = params[-1]
        remains_params = params[1:-1]
        choosed_params = remains_params[1::step]
        choosed_params = [first_param] + choosed_params + [last_param]
        return list(set(choosed_params))

    def check_param(self, param, value, types):
        if isinstance(value, list) and len(value):
            for p in value:
                if not isinstance(p, types[param]) and p is not None:
                    raise Exception('The value of the "{0}" parameter must be a "{1}", byt was "{2}"'.
                                    format(param, types[param], type(p)))
        else:
            raise Exception('The value of the "{0}" parameter must be a non-empty list'.format(param))