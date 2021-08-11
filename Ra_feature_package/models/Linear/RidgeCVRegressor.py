import os
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from Ra_feature_package.Errors import Errors
from Ra_feature_package.models.static_methods import *


class RCVRegressor:
    def __init__(self,
                 task: pd.DataFrame,
                 target: pd.DataFrame,
                 train_split: int,
                 show: bool = False):
        """
        This method is the initiator of the RCVRegressor class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__text_name = "RidgeCVRegressor"
        self.__default_param_types = {'alphas': type(np.ndarray),
                                      'fit_intercept': bool,
                                      'normalize': bool,
                                      'scoring': str,
                                      'cv': int,
                                      'gcv_mode': str,
                                      'store_cv_values': bool,
                                      'alpha_per_target': bool}

        self.__default_param = {'alphas': np.array([0.1, 1.0, 10.0]),
                                'fit_intercept': True,
                                'normalize': False,
                                'scoring': None,
                                'cv': None,
                                'gcv_mode': 'auto',
                                'store_cv_values': False,
                                'alpha_per_target': False}

        count = len(task.keys()) + 1
        self.__default_params = {'alphas': [np.array(list(x / 10 for x in range(1, 101)))],
                                 'fit_intercept': [True, False],
                                 'normalize': [True, False],
                                 'scoring': [None],
                                 'cv': [None],
                                 'gcv_mode': ['auto', 'svd', 'eigen'],
                                 'store_cv_values': [True, False],
                                 'alpha_per_target': [True, False]}
        self.__locked_params = ['fit_intercept', 'normalize', 'scoring', 'gcv_mode',
                                'store_cv_values', 'alpha_per_target']
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
        return f"'<Ra.{RCVRegressor.__name__} model>'"

    def __repr__(self):
        return f"'<Ra.{RCVRegressor.__name__} model>'"

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def fit(self,
            param_dict: Dict[str, int or str] = None,
            grid_params: bool = False,
            show: bool = False):
        f"""
        This method trains the model {self.__text_name}, it is possible to use the parameters from "fit_grid"
        :param param_dict: The parameter of the hyperparameter grid that we check
        :param grid_params: The switcher which is responsible for the ability to use all the ready-made parameters
         from avia for training
        """
        if grid_params and param_dict is None:
            self.model = RidgeCV(alphas=self.__grid_best_params['alphas'],
                                 fit_intercept=self.__grid_best_params['fit_intercept'],
                                 normalize=self.__grid_best_params['normalize'],
                                 scoring=self.__grid_best_params['scoring'],
                                 cv=self.__grid_best_params['cv'],
                                 gcv_mode=self.__grid_best_params['gcv_mode'],
                                 store_cv_values=self.__grid_best_params['store_cv_values'],
                                 alpha_per_target=self.__grid_best_params['alpha_per_target'])
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

            self.model = RidgeCV(alphas=model_params['alphas'],
                                 fit_intercept=model_params['fit_intercept'],
                                 normalize=model_params['normalize'],
                                 scoring=model_params['scoring'],
                                 cv=model_params['cv'],
                                 gcv_mode=model_params['gcv_mode'],
                                 store_cv_values=model_params['store_cv_values'],
                                 alpha_per_target=model_params['alpha_per_target'])

        elif not grid_params and param_dict is None:
            self.model = RidgeCV()
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        if show:
            print(f"Learning {self.__text_name}...")
        self.model.fit(self.__X_train, self.__Y_train.values.ravel())
        self.__is_model_fit = True

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 count: int = 0,
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
        model = RidgeCV()
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
        f"""
        This method calculates the "roc_auc_score" for the {self.__text_name} on the test data
        :return: roc_auc_score
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_roc_auc_score(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"roc_auc_score\" error")
        return error

    def get_mean_squared_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        :return: mean_squared_error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"mean_squared_error\" error")
        return error

    def get_mean_absolute_error(self) -> float:
        """
        This method calculates the "mean_absolute_error" for the {self.text_name} on the test data
        :return: mean_absolute_error
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_absolute_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            print("An error occurred when calculating the \"mean_absolute_error\" error")
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
        model = RidgeCV()
        model.fit(self.__X_train, self.__Y_train)
        time_end = time.time()
        return time_end - time_start