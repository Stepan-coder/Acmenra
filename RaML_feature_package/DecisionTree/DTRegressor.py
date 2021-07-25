import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, List


class DTRegressor:
    def __init__(self, task: pd.DataFrame, target: pd.DataFrame, train_split: int, show: bool = False):
        """
        This method is the initiator of the DTRegressor class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.is_grid_fit = False
        self.show = show
        self.importance = {}
        self.is_model_fit = False
        self.is_grid_fit = False
        self.grid_best_params = None
        self.keys = task.keys()
        self.keys_len = len(task.keys())
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(task,
                                                                                target,
                                                                                train_size=train_split,
                                                                                random_state=13)
        self.param_types = {'criterion': str,
                            'splitter': str,
                            'max_depth': int,
                            'min_samples_split': int,
                            'min_samples_leaf': int,
                            'min_weight_fraction_leaf': float,
                            'max_features': str,
                            'max_leaf_nodes': int,
                            'min_impurity_decrease': float,
                            'min_impurity_split': float,
                            'ccp_alpha': float
                            }

        self.default_param = {'criterion': "mse",
                              'splitter': "best",
                              'max_depth': None,
                              'min_samples_split': None,
                              'min_samples_leaf': None,
                              'min_weight_fraction_leaf': 0.,
                              'max_features': None,
                              'max_leaf_nodes': None,
                              'min_impurity_decrease': 0.,
                              'min_impurity_split': None,
                              'ccp_alpha': 0.0
                              }
        self.default_params = {'criterion': ["mse", "friedman_mse", "mae", "poisson"],
                               'splitter': ["best", "random"],
                               'max_depth': [i for i in range(1, self.keys_len + 1)] + [None],
                               'min_samples_split': [i for i in range(2, self.keys_len + 1)],
                               'min_samples_leaf': [i for i in range(1, self.keys_len + 1)],
                               'max_features': ['sqrt', 'auto', 'log2', None],
                               'max_leaf_nodes': [None],
                               'min_impurity_decrease': [0.],
                               'min_impurity_split': [None],
                               'ccp_alpha': [0.0]
                               }

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 step: int = 1,
                 cross_validation: int = 3):
        if params_dict is not None:
            for param in params_dict:
                if param not in self.default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                self.check_param(param, params_dict[param], self.param_types)
                self.default_params[param] = params_dict[param]
        print("Learning GridSearch DecisionTreeClassifier...")
        if self.show:
            self.show_grid_params(self.default_params)
        dtc = DecisionTreeRegressor(random_state=13)
        grid = GridSearchCV(dtc, self.default_params, cv=cross_validation)
        grid.fit(self.X_train, self.Y_train)
        self.grid_best_params = grid.best_params_
        self.is_grid_fit = True

    def fit(self, param_dict: Dict[str, int or str] = None, grid_params: bool = False):
        if grid_params and param_dict is None:
            self.dt = DecisionTreeRegressor(criterion=self.grid_best_params['criterion'],
                                            splitter=self.grid_best_params['splitter'],
                                            max_depth=self.grid_best_params['max_depth'],
                                            min_samples_split=self.grid_best_params['min_samples_split'],
                                            min_samples_leaf=self.grid_best_params['min_samples_leaf'],
                                            min_weight_fraction_leaf=self.grid_best_params['min_weight_fraction_leaf'],
                                            max_features=self.grid_best_params['max_features'],
                                            random_state=13,
                                            max_leaf_nodes=self.grid_best_params['max_leaf_nodes'],
                                            min_impurity_decrease=self.grid_best_params['min_impurity_decrease'],
                                            min_impurity_split=self.grid_best_params['min_impurity_split'],
                                            ccp_alpha=self.grid_best_params['ccp_alpha'])

        elif not grid_params and param_dict is not None:
            for param in param_dict:
                if param not in self.default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                self.check_param(param, param_dict[param], self.param_types)
                self.default_param[param] = param_dict[param]
            self.dt = DecisionTreeRegressor(criterion=param_dict['criterion'],
                                            splitter=param_dict['splitter'],
                                            max_depth=param_dict['max_depth'],
                                            min_samples_split=param_dict['min_samples_split'],
                                            min_samples_leaf=param_dict['min_samples_leaf'],
                                            min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                            max_features=param_dict['max_features'],
                                            random_state=13,
                                            max_leaf_nodes=param_dict['max_leaf_nodes'],
                                            min_impurity_decrease=param_dict['min_impurity_decrease'],
                                            min_impurity_split=param_dict['min_impurity_split'],
                                            ccp_alpha=param_dict['ccp_alpha'])
        elif not grid_params and param_dict is None:
            self.dt = DecisionTreeRegressor()
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        print("Learning DecisionTreeClassifier...")
        self.dt.fit(self.X_train, self.Y_train)
        self.is_model_fit = True

    def get_best_params(self,):
        if self.is_grid_fit:
            return self.grid_best_params
        else:
            raise Exception('At first you need to learn grid')

    def get_feature_importances(self):
        if not self.is_model_fit:
            raise Exception('You haven"t trained the DecisionTreeClassifier yet')
        for index in range(len(self.dt.feature_importances_)):
            self.importance[self.keys[index]] = self.dt.feature_importances_[index]
        return {k: v for k, v in sorted(self.importance.items(), key=lambda item: item[1], reverse=True)}

    def get_roc_auc_score(self):
        if not self.is_model_fit:
            raise Exception('You haven"t trained the DecisionTreeClassifier yet')
        return roc_auc_score(self.y_test, self.dt.predict(self.x_test))

    def get_mean_squared_error(self):
        if not self.is_model_fit:
            raise Exception('You haven"t trained the DecisionTreeClassifier yet')
        return mean_squared_error(self.y_test, self.dt.predict(self.x_test))

    def get_mean_absolute_error(self):
        if not self.is_model_fit:
            raise Exception('You haven"t trained the DecisionTreeClassifier yet')
        return mean_absolute_error(self.y_test, self.dt.predict(self.x_test))

    def show_grid_params(self, params):
        count_elements = []
        multiply = 1
        for param in params:
            print("Param({0}[{2}]): {1}".format(param, params[param], len(params[param])))
            count_elements.append(len(params[param]))
        for ce in count_elements:
            multiply *= ce
        print("Total({1}): {0}".format(" X ".join([str(e) for e in count_elements]), multiply))

    @staticmethod
    def get_choosed_params(params, step):
        first_param = params[0]
        last_param = params[-1]
        remains_params = params[1:-1]
        choosed_params = remains_params[1::step]
        choosed_params = [first_param] + choosed_params + [last_param]
        choosed_params = list(set(choosed_params))
        choosed_params.sort()
        return choosed_params

    @staticmethod
    def check_param(param, value, types):
        if isinstance(value, list) and len(value):
            for p in value:
                if not isinstance(p, types[param]) and p is not None:
                    raise Exception(f"The value of the {param} parameter must be a {types[param]}, byt was {type(p)}")
        else:
            raise Exception(f"The value of the {param} parameter must be a non-empty list")
