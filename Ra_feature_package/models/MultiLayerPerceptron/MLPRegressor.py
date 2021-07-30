import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MultiLayerPerceptronRegressor
from Ra_feature_package.Errors import Errors
from Ra_feature_package.models.static_methods import *


class MLPRegressor:
    def __init__(self,
                 task: pd.DataFrame,
                 target: pd.DataFrame,
                 train_split: int,
                 show: bool = False):
        """
        This method is the initiator of the MLPRegressor class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__text_name = "MultiLayerPerceptronRegressor"
        self.__default_param_types = {'loss': str,
                                      'penalty': str,
                                      'alpha': float,
                                      'l1_ratio': float,
                                      'fit_intercept': bool,
                                      'max_iter': int,
                                      'tol': float,
                                      'shuffle': bool,
                                      'verbose': int,
                                      'epsilon': float,
                                      'learning_rate': float,
                                      'eta0': float,
                                      'power_t': float,
                                      'early_stopping': bool,
                                      'validation_fraction': float,
                                      'n_iter_no_change': int,
                                      'warm_start': bool,
                                      'average': bool}

        self.__default_param = {'loss': 'squared_loss',
                                'penalty': 'l2',
                                'alpha': 0.0001,
                                'l1_ratio': 0.15,
                                'fit_intercept': True,
                                'max_iter': 1000,
                                'tol': 1e-3,
                                'shuffle': True,
                                'verbose': 0,
                                'epsilon': 0.1,
                                'learning_rate': 'invscaling',
                                'eta0': 0.01,
                                'power_t': 0.25,
                                'early_stopping': False,
                                'validation_fraction': 0.1,
                                'n_iter_no_change': 5,
                                'warm_start': False,
                                'average': False}

        # Доделать
        count = len(task.keys()) + 1
        self.__default_params = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                                 'penalty': ['l2', 'l1', 'elasticnet'],
                                 'alpha': conf_params(min_val=0, max_val=count*0.00005, count=count, ltype=float),
                                 'l1_ratio': conf_params(min_val=0, max_val=1, count=count, ltype=float),
                                 'fit_intercept': [True, False],
                                 'max_iter': conf_params(min_val=2, count=count*100, ltype=int),
                                 'tol': conf_params(min_val=2, max_val=count*0.00005, count=count, ltype=float),
                                 'shuffle': [True, False],
                                 'verbose': conf_params(min_val=2, count=count, ltype=int),
                                 'epsilon': conf_params(min_val=0, max_val=count*0.05, count=count, ltype=float),
                                 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                 'eta0': conf_params(min_val=2, max_val=count*0.005, count=count, ltype=float),
                                 'power_t': conf_params(min_val=0, max_val=count*0.025, count=count,  ltype=float),
                                 'early_stopping': [True, False],
                                 'validation_fraction': conf_params(min_val=0, max_val=1, count=count, ltype=float),
                                 'n_iter_no_change': conf_params(min_val=2, count=count, ltype=int),
                                 'warm_start': [True, False],
                                 'average': [True, False]}
        self.__locked_params = ['loss', 'penalty', 'fit_intercept', 'shuffle', 'learning_rate', 'early_stopping',
                                'warm_start', 'average']
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
        return f"'<Ra.{MLPRegressor.__name__} model>'"

    def __repr__(self):
        return f"'<Ra.{MLPRegressor.__name__} model>'"

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def fit(self,
            param_dict: Dict[str, int or str] = None,
            grid_params: bool = False):
        f"""
        This method trains the model {self.__text_name}, it is possible to use the parameters from "fit_grid"
        :param param_dict: The parameter of the hyperparameter grid that we check
        :param grid_params: The switcher which is responsible for the ability to use all the ready-made parameters
         from avia for training
        """
        if grid_params and param_dict is None:
            self.model = MultiLayerPerceptronRegressor(loss=self.__grid_best_params['loss'],
                                                       penalty=self.__grid_best_params['penalty'],
                                                       alpha=self.__grid_best_params['alpha'],
                                                       l1_ratio=self.__grid_best_params['l1_ratio'],
                                                       fit_intercept=self.__grid_best_params['fit_intercept'],
                                                       max_iter=self.__grid_best_params['max_iter'],
                                                       tol=self.__grid_best_params['tol'],
                                                       shuffle=self.__grid_best_params['shuffle'],
                                                       verbose=self.__grid_best_params['verbose'],
                                                       epsilon=self.__grid_best_params['epsilon'],
                                                       learning_rate=self.__grid_best_params['learning_rate'],
                                                       eta0=self.__grid_best_params['eta0'],
                                                       power_t=self.__grid_best_params['power_T'],
                                                       early_stopping=self.__grid_best_params['early_stopping'],
                                                       validation_fraction=self.__grid_best_params['validation_fraction'],
                                                       n_iter_no_change=self.__grid_best_params['n_iter_no_change'],
                                                       warm_start=self.__grid_best_params['warm_start'],
                                                       average=self.__grid_best_params['average'],
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
            self.model = MultiLayerPerceptronRegressor(loss=model_params['loss'],
                                                       penalty=model_params['penalty'],
                                                       alpha=model_params['alpha'],
                                                       l1_ratio=model_params['l1_ratio'],
                                                       fit_intercept=model_params['fit_intercept'],
                                                       max_iter=model_params['max_iter'],
                                                       tol=model_params['tol'],
                                                       shuffle=model_params['shuffle'],
                                                       verbose=model_params['verbose'],
                                                       epsilon=model_params['epsilon'],
                                                       learning_rate=model_params['learning_rate'],
                                                       eta0=model_params['eta0'],
                                                       power_t=model_params['power_T'],
                                                       early_stopping=model_params['early_stopping'],
                                                       validation_fraction=model_params['validation_fraction'],
                                                       n_iter_no_change=model_params['n_iter_no_change'],
                                                       warm_start=model_params['warm_start'],
                                                       average=model_params['average'],
                                                       random_state=13)
        elif not grid_params and param_dict is None:
            self.model = MultiLayerPerceptronRegressor()
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        print(f"Learning {self.__text_name}...")
        self.model.fit(self.__X_train, self.__Y_train.values.ravel())
        self.__is_model_fit = True

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 count: int = 1,
                 cross_validation: int = 3):
        """
        This method uses iteration to find the best hyperparameters for the model and trains the model using them
        :param params_dict: The parameter of the hyperparameter grid that we check
        :param count: The step with which to return the values
        :param cross_validation: The number of sections into which the dataset will be divided for training
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
            model_params[param] = get_choosed_params(model_params[param], count=count)

        if self.__show:
            print(f"Learning GridSearch {self.__text_name}...")
            show_grid_params(model_params)
        model = MultiLayerPerceptronRegressor(random_state=13)
        grid = GridSearchCV(model, model_params, cv=cross_validation)
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
            plt.savefig(f"{save_path}\\Test predict {self.__text_name}.png")
        if show:
            plt.show()
        plt.close()



