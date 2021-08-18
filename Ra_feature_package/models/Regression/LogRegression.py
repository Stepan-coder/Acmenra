import os
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List
from prettytable import PrettyTable
from Ra_feature_package.Errors import Errors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Ra_feature_package.models.static_methods import *


class LogRegressor:
    def __init__(self,
                 task: pd.DataFrame,
                 target: pd.DataFrame,
                 train_split: int,
                 show: bool = False):
        """
        This method is the initiator of the LogisticRegression class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__text_name = "LogisticRegression"
        self.__default_param_types = {'penalty': str,
                                      'dual': bool,
                                      'tol': float,
                                      'C': float,
                                      'fit_intercept': bool,
                                      'intercept_scaling': float,
                                      'class_weight': dict,
                                      'solver': str,
                                      'max_iter': int,
                                      'multi_class': str,
                                      'warm_start': bool,
                                      'l1_ratio': float}

        self.__default_param = {'penalty': "l2",
                                'dual': False,
                                'tol': 1e-4,
                                'C': 1.0,
                                'fit_intercept': True,
                                'intercept_scaling': 1,
                                'class_weight': None,
                                'solver': "lbfgs",
                                'max_iter': 100,
                                'multi_class': 'auto',
                                'warm_start': False,
                                'l1_ratio': None}

        count = len(task.keys()) + 1
        self.__default_params = {'penalty': ["l2", "elasticnet", "none"],
                                 'dual': [False],
                                 'tol': [1e-4],  # Лучше не трогать
                                 'C': [1.0],  # Лучше не трогать
                                 'fit_intercept': [True, False],
                                 'intercept_scaling': [1],
                                 'class_weight': [None],  # Лучше не трогать
                                 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                 'max_iter': conf_params(min_val=2, max_val=count, count=count, ltype=int),
                                 'multi_class': ['auto', 'ovr', 'multinomial'],
                                 'warm_start': [True, False],
                                 'l1_ratio': [None]}
        self.__locked_params = ['penalty', 'fit_intercept', 'solver', 'multi_class', 'warm_start']
        self.__importance = {}
        self.__is_model_fit = False
        self.__is_grid_fit = False

        self.__show = show
        self.model = None
        self.__grid_best_params = None
        self.__keys = task.keys()
        self.__keys_len = len(task.keys())
        self.__X_train, self.__x_test, self.__Y_train, self.__y_test = train_test_split(task,
                                                                                        target,
                                                                                        train_size=train_split,
                                                                                        random_state=13)

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

    def __repr__(self):
        return f"'<Ra.{LogRegressor.__name__} model>'"

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def fit(self,
            param_dict: Dict[str, int or str] = None,
            grid_params: bool = False,
            n_jobs: int = 1,
            verbose: int = 0):
        f"""
        This method trains the model {self.__text_name}, it is possible to use the parameters from "fit_grid"
        :param param_dict: The parameter of the hyperparameter grid that we check
        :param grid_params: The switcher which is responsible for the ability to use all the ready-made parameters
         from avia for training
        :param n_jobs: The number of jobs to run in parallel.
        :param verbose: Learning-show param
        """
        if grid_params and param_dict is None:
            self.model = LogisticRegression(penalty=self.__grid_best_params['penalty'],
                                            dual=self.__grid_best_params['dual'],
                                            tol=self.__grid_best_params['tol'],
                                            C=self.__grid_best_params['C'],
                                            fit_intercept=self.__grid_best_params['fit_intercept'],
                                            intercept_scaling=self.__grid_best_params['intercept_scaling'],
                                            class_weight=self.__grid_best_params['class_weight'],
                                            solver=self.__grid_best_params['solver'],
                                            max_iter=self.__grid_best_params['max_iter'],
                                            multi_class=self.__grid_best_params['multi_class'],
                                            warm_start=self.__grid_best_params['warm_start'],
                                            l1_ratio=self.__grid_best_params['l1_ratio'],
                                            n_jobs=n_jobs,
                                            verbose=verbose,
                                            random_state=13)
        elif not grid_params and param_dict is not None:
            model_params = self.__default_param
            for param in param_dict:
                if param not in self.__default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                check_param(param,
                            param_dict[param],
                            self.__default_param_types[param],
                            type(self.__default_param[param]))
                model_params[param] = param_dict[param]

            self.model = LogisticRegression(penalty=model_params['penalty'],
                                            dual=model_params['dual'],
                                            tol=model_params['tol'],
                                            C=model_params['C'],
                                            fit_intercept=model_params['fit_intercept'],
                                            intercept_scaling=model_params['intercept_scaling'],
                                            class_weight=model_params['class_weight'],
                                            solver=model_params['solver'],
                                            max_iter=model_params['max_iter'],
                                            multi_class=model_params['multi_class'],
                                            warm_start=model_params['warm_start'],
                                            l1_ratio=model_params['l1_ratio'],
                                            n_jobs=n_jobs,
                                            verbose=verbose,
                                            random_state=13)
        elif not grid_params and param_dict is None:
            self.model = LogisticRegression(n_jobs=n_jobs,
                                            verbose=verbose,
                                            random_state=13)
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        print(f"Learning {self.__text_name}...")
        self.model.fit(self.__X_train, self.__Y_train.values.astype(float))
        self.__is_model_fit = True

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 count: int = 1,
                 cross_validation: int = 3,
                 grid_n_jobs: int = 1):
        """
        This method uses iteration to find the best hyperparameters for the model and trains the model using them
        :param params_dict: The parameter of the hyperparameter grid that we check
        :param count: The step with which to return the values
        :param cross_validation: The number of sections into which the dataset will be divided for training
        :param grid_n_jobs: The number of jobs to run in parallel.
        """
        model_params = self.__default_params
        if params_dict is not None:
            for param in params_dict:
                if param not in self.__default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                check_param(grid_param=param,
                            value=params_dict[param],
                            param_type=self.__default_param_types[param],
                            setting_param_type=type(self.__default_params[param]))
                model_params[param] = params_dict[param]

        for param in [p for p in model_params if p not in self.__locked_params]:
            if count != 0:
                model_params[param] = get_choosed_params(model_params[param],
                                                         count=count,
                                                         ltype=self.__default_param_types[param])
            else:
                model_params[param] = [self.__default_param[param]]

        if self.__show:
            print(f"Learning GridSearch {self.__text_name}...")
            show_grid_params(params=model_params,
                             locked_params=self.__locked_params,
                             single_model_time=self.__get_default_model_fit_time(),
                             n_jobs=grid_n_jobs)
        model = LogisticRegression(n_jobs=1,
                                   verbose=0,
                                   random_state=13)
        grid = GridSearchCV(model,
                            model_params,
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
        return self.__locked_params

    def get_non_locked_params(self) -> List[str]:
        """
        :return: This method return the list of non locked params
        """
        return [p for p in self.__default_params if p not in self.__locked_params]

    def get_default_param_types(self) -> dict:
        """
        :return: This method return default model param types
        """
        return self.__default_param_types

    def get_default_param_values(self) -> dict:
        """
        :return: This method return default model param values
        """
        return self.__default_param

    def get_default_grid_param_values(self) -> dict:
        """
        :return: This method return default model param values for grid search
        """
        return self.__default_params

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

    def get_predict_text_plt(self,
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

    def __get_default_model_fit_time(self) -> float:
        """
        This method return time of fit model with defualt params
        :return: time of fit model with defualt params
        """
        time_start = time.time()
        model = LogisticRegression(random_state=13)
        model.fit(self.__X_train, self.__Y_train)
        time_end = time.time()
        return time_end - time_start