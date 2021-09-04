import os
import math
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List
from prettytable import PrettyTable
from sklearn.linear_model import Lasso
from Ra_feature_package.Errors import Errors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from Ra_feature_package.models.static_methods import *
from Ra_feature_package.models.Param import *


class LassoRegressor:
    def __init__(self,
                 task: pd.DataFrame or list = None,
                 target: pd.DataFrame or list = None,
                 train_split: int = None,
                 show: bool = False):
        """
        This method is the initiator of the LassoRegressor class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__text_name = "LassoRegressor"
        self.__importance = {}
        self.__is_dataset_set = False
        self.__is_model_fit = False
        self.__is_grid_fit = False

        self.__show = show
        self.model = None
        self.__grid_best_params = None
        self.__keys = None
        self.__keys_len = None
        self.__default = None
        self.__X_train = None
        self.__x_test = None
        self.__Y_train = None
        self.__y_test = None

        if task is not None and target is not None and train_split is not None:
            self.set_params(task=task,
                            target=target,
                            train_split=train_split,
                            show=show)

    def __str__(self):
        table = PrettyTable()
        table.title = f"{'Untrained ' if not self.__is_model_fit else ''}\"{self.__text_name}\" model"
        table.field_names = ["Error", "Result"]
        if self.__is_model_fit:
            table.add_row(["ROC AUC score", self.get_roc_auc_score()])
            table.add_row(["R-Squared_error", self.get_r_squared_error()])
            table.add_row(["Mean Absolute Error", self.get_mean_absolute_error()])
            table.add_row(["Mean Squared Error", self.get_mean_squared_error()])
            table.add_row(["Root Mean Squared Error", self.get_root_mean_squared_error()])
            table.add_row(["Median Absolute Error", self.get_median_absolute_error()])
        return str(table)

    def __repr__(self):
        table = PrettyTable()
        table.title = f"{'Untrained ' if not self.__is_model_fit else ''}\"{self.__text_name}\" model"
        table.field_names = ["Error", "Result"]
        if self.__is_model_fit:
            table.add_row(["ROC AUC score", self.get_roc_auc_score()])
            table.add_row(["R-Squared_error", self.get_r_squared_error()])
            table.add_row(["Mean Absolute Error", self.get_mean_absolute_error()])
            table.add_row(["Mean Squared Error", self.get_mean_squared_error()])
            table.add_row(["Root Mean Squared Error", self.get_root_mean_squared_error()])
            table.add_row(["Median Absolute Error", self.get_median_absolute_error()])
        return str(table)

    def set_params(self,
                   task: pd.DataFrame or list,
                   target: pd.DataFrame or list,
                   train_split: int,
                   show: bool = False):
        count = len(task.keys()) + 1
        self.__default = {'alpha': Param(ptype=[float],
                                         def_val=1.0,
                                         def_vals=[1.0]),
                          'fit_intercept': Param(ptype=[bool],
                                                 def_val=True,
                                                 def_vals=[True, False],
                                                 is_locked=True),
                          'normalize': Param(ptype=[bool],
                                             def_val=False,
                                             def_vals=[True, False],
                                             is_locked=True),
                          'precompute': Param(ptype=[bool],
                                              def_val=False,
                                              def_vals=[True, False],
                                              is_locked=True),
                          'copy_X': Param(ptype=[bool],
                                          def_val=True,
                                          def_vals=[True, False],
                                          is_locked=True),
                          'max_iter': Param(ptype=[int],
                                            def_val=1000,
                                            def_vals=conf_params(min_val=250,
                                                                 max_val=count,
                                                                 count=count,
                                                                 ltype=int)),
                          'tol': Param(ptype=[float],
                                       def_val=1e-4,
                                       def_vals=[1e-4]),

                          'warm_start': Param(ptype=[bool],
                                              def_val=False,
                                              def_vals=[True, False],
                                              is_locked=True),
                          'positive': Param(ptype=[bool],
                                            def_val=False,
                                            def_vals=[True, False],
                                            is_locked=True),
                          'selection': Param(ptype=[str],
                                             def_val='cyclic',
                                             def_vals=['cyclic', 'random'],
                                             is_locked=True)}
        self.__show = show
        self.__keys = task.keys()
        self.__keys_len = len(task.keys())
        self.__X_train, self.__x_test, self.__Y_train, self.__y_test = train_test_split(task,
                                                                                        target,
                                                                                        train_size=train_split,
                                                                                        random_state=13)
        self.__is_dataset_set = True

    def predict(self, data: pd.DataFrame):
        """
        This method predicting values on data
        :param data:
        """
        if not self.__is_model_fit:
            raise Exception('At first you need to learn model!')
        return self.model.predict(data)

    def fit(self,
            param_dict: Dict[str, int or str] = None,
            grid_params: bool = False,
            n_jobs: int = 1,
            verbose: int = 0):
        """
        This method trains the model {self.__text_name}, it is possible to use the parameters from "fit_grid"
        :param param_dict: The parameter of the hyperparameter grid that we check
        :param grid_params: The switcher which is responsible for the ability to use all the ready-made parameters
         from avia for training
        :param n_jobs: The number of jobs to run in parallel.
        :param verbose: Learning-show param
        """
        if not self.__is_dataset_set:
            raise Exception('At first you need set dataset!')
        if grid_params and param_dict is None:
            self.model = Lasso(**self.__grid_best_params,
                               random_state=13)
        elif not grid_params and param_dict is not None:
            model_params = self.get_default_grid_param_values()
            for param in param_dict:
                if param not in self.__default.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                check_param_value(grid_param=param,
                                  value=param_dict[param],
                                  param_type=self.__default[param].ptype)
                model_params[param] = param_dict[param]
            self.model = Lasso(**model_params,
                               random_state=13)
        elif not grid_params and param_dict is None:
            self.model = Lasso(random_state=13)
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        if self.__show:
            print(f"Learning {self.__text_name}...")
        self.model.fit(self.__X_train, self.__Y_train.values.ravel())
        self.__is_model_fit = True

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 count: int = 0,  # Это имеется в виду из пользовательской сетки
                 cross_validation: int = 2,
                 grid_n_jobs: int = 1):
        """
        This method uses iteration to find the best hyperparameters for the model and trains the model using them
        :param params_dict: The parameter of the hyperparameter grid that we check
        :param count: The step with which to return the values
        :param cross_validation: The number of sections into which the dataset will be divided for training
        :param grid_n_jobs: The number of jobs to run in parallel.
        """
        if not self.__is_dataset_set:
            raise Exception('At first you need set dataset!')
        model_params = self.get_default_grid_param_values()
        if params_dict is not None:
            for param in params_dict:
                if param not in self.__default.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                check_params_list(grid_param=param,
                                  value=params_dict[param],
                                  param_type=self.__default[param].ptype)
                model_params[param] = params_dict[param]

        for param in [p for p in model_params if not self.__default[p].is_locked]:
            if count > 0:  # Если идёт поиск по полной сетке
                if params_dict is None:  # Если пользователь не задал параметры
                    model_params[param] = [self.__default[param].def_val] + \
                                          get_choosed_params(params=model_params[param],
                                                             count=count,
                                                             ltype=self.__default[param].ptype)
                else:
                    if param not in params_dict:  # Если пользователь, не трогал это поле
                        model_params[param] = [self.__default[param].def_val] + \
                                              get_choosed_params(params=model_params[param],
                                                                 count=count,
                                                                 ltype=self.__default[param].ptype)
                    else:
                        model_params[param] = model_params[param]
            else:
                if params_dict is None:
                    model_params[param] = [self.__default[param].def_val]
                else:
                    if param not in params_dict:  # Если пользователь, не трогал это поле
                        model_params[param] = [self.__default[param].def_val]
                    else:
                        model_params[param] = model_params[param]
        if self.__show:
            print(f"Learning GridSearch {self.__text_name}...")
            show_grid_params(params=model_params,
                             locked_params=self.get_locked_params(),
                             single_model_time=self.__get_default_model_fit_time(),
                             n_jobs=grid_n_jobs)
        model = Lasso(random_state=13)
        grid = GridSearchCV(estimator=model,
                            param_grid=model_params,
                            cv=cross_validation,
                            n_jobs=grid_n_jobs,
                            scoring='neg_mean_absolute_error')
        grid.fit(self.__X_train, self.__Y_train.values.ravel())
        self.__grid_best_params = grid.best_params_
        self.__is_grid_fit = True

    def get_locked_params(self) -> List[str]:
        """
        :return: This method return the list of locked params
        """
        return [p for p in self.__default if self.__default[p].is_locked]

    def get_non_locked_params(self) -> List[str]:
        """
        :return: This method return the list of non locked params
        """
        return [p for p in self.__default if not self.__default[p].is_locked]

    def get_default_param_types(self) -> dict:
        """
        :return: This method return default model param types
        """
        default_param_types = {}
        for default in self.__default:
            default_param_types[default] = self.__default[default].ptype
        return default_param_types

    def get_default_param_values(self) -> dict:
        """
        :return: This method return default model param values
        """
        default_param_values = {}
        for default in self.__default:
            default_param_values[default] = self.__default[default].def_val
        return default_param_values

    def get_default_grid_param_values(self) -> dict:
        """
        :return: This method return default model param values for grid search
        """
        default_param_values = {}
        for default in self.__default:
            default_param_values[default] = self.__default[default].def_vals
        return default_param_values

    def get_is_model_fit(self) -> bool:
        f"""
        This method return flag is_model_fit
        :return: is_model_fit
        """
        return self.__is_model_fit

    def get_is_grid_fit(self) -> bool:
        f"""
        This method return flag get_is_grid_fit
        :return: get_is_grid_fit
        """
        return self.__is_grid_fit

    def get_grid_best_params(self) -> dict:
        """
        This method return the dict of best params for this model
        :return: dict of best params for this model
        """
        if self.__is_grid_fit:
            return self.__grid_best_params
        else:
            raise Exception('At first you need to learn grid')

    def get_feature_importance(self) -> dict:
        """
        This method return dict of feature importance where key is the column of input dataset, and value is importance
        of this column
        :return: dict of column importance
        """
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        for index in range(len(self.model.feature_importances_)):
            self.__importance[self.__keys[index]] = self.model.feature_importances_[index]
        return {k: v for k, v in sorted(self.__importance.items(), key=lambda item: item[1], reverse=True)}

    def copy(self):
        """
        This method return copy of this class
        :return: copy of this class
        """
        return copy.copy(self)

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

    def get_root_mean_squared_error(self) -> float:
        """
        This method calculates the "root_mean_squared_error" for the {self.text_name} on the test data
        :return: Root Mean Squared Error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_root_mean_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"Root Mean Squared Error\" error")
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

    def get_predict_test_plt(self,
                             save_path: str = None,
                             show: bool = False):
        """
        This method automates the display/saving of a graph of prediction results with a real graph
        :param save_path: The path to save the graph on
        :param show: The parameter responsible for displaying the plot of prediction
        """
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        values = [i for i in range(len(self.__x_test))]
        plt.title(f'Predict {self.__text_name} at test data')
        plt.plot(values, self.__y_test, 'g-', label='test')
        plt.plot(values, self.model.predict(self.__x_test), 'r-', label='predict')
        plt.legend(loc='best')
        if save_path is not None:
            if not os.path.exists(save_path):  # Надо что то с путём что то адекватное придумать
                raise Exception("The specified path was not found!")
            plt.savefig(os.path.join(save_path, f"Test predict {self.__text_name}.png"))
        if show:
            plt.show()
        plt.close()

    def set_train_test(self, X_train, x_test, Y_train, y_test):
        self.__X_train, self.__x_test, self.__Y_train, self.__y_test = X_train, x_test, Y_train, y_test
        self.__is_dataset_set = True

    def __get_default_model_fit_time(self) -> float:
        """
        This method return time of fit model with defualt params
        :return: time of fit model with defualt params
        """
        time_start = time.time()
        model = Lasso(random_state=13)
        model.fit(self.__X_train, self.__Y_train)
        time_end = time.time()
        return time_end - time_start