import os
import time
import copy
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict
from prettytable import PrettyTable
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from RA.Errors import Errors
from RA.models.Param import *
from RA.models.static_methods import *


class ABoostClassifier:
    def __init__(self,
                 task: pd.DataFrame or list = None,
                 target: pd.DataFrame or list = None,
                 train_split: int = None,
                 show: bool = False):
        """
        This method is the initiator of the AdaBoostClassifier class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__text_name = "AdaBoostClassifier"
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
        self.set_params(count=25)
        if task is not None and target is not None and train_split is not None:
            self.set_data(task=task,
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

    def set_params(self, count: int) -> None:
        """
        This method sets the parameters for the training grid
        :param count: The number of elements in the grid
        """
        self.__default = {'base_estimator': Param(ptype=[type(None)],
                                                  def_val=None,
                                                  def_vals=[None]),
                          'n_estimators': Param(ptype=[int],
                                                def_val=50,
                                                def_vals=conf_params(min_val=1,
                                                                     max_val=count * 50,
                                                                     count=count,
                                                                     ltype=int)),
                          'learning_rate': Param(ptype=[float],
                                                 def_val=1.0,
                                                 def_vals=[1.0]),
                          'algorithm': Param(ptype=[str],
                                             def_val='SAMME.R',
                                             def_vals=['SAMME', 'SAMME.R'],
                                             is_locked=True)}

    def set_data(self,
                 task: pd.DataFrame or list,
                 target: pd.DataFrame or list,
                 train_split: int,
                 show: bool = False) -> None:
        """
        This method passes data to the class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.__show = show
        self.__keys = task.keys()
        self.__keys_len = len(task.keys())
        self.__X_train, self.__x_test, self.__Y_train, self.__y_test = train_test_split(task,
                                                                                        target,
                                                                                        train_size=train_split,
                                                                                        random_state=13)
        self.__is_dataset_set = True

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
            self.model = AdaBoostClassifier(**self.__grid_best_params,
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
            self.model = AdaBoostClassifier(**model_params,
                                            random_state=13)
        elif not grid_params and param_dict is None:
            self.model = AdaBoostClassifier(random_state=13)
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
        for param in model_params:
            model_params[param] = list(set(model_params[param]))
            has_none = None in model_params[param]
            model_params[param] = [p for p in model_params[param] if p is not None]
            try:
                model_params[param].sort()
            except:
                pass
            if has_none:
                model_params[param].append(None)
        if self.__show:
            print(f"Learning GridSearch {self.__text_name}...")
            show_grid_params(params=model_params,
                             locked_params=self.get_locked_params_names(),
                             single_model_time=self.__get_default_model_fit_time(),
                             n_jobs=grid_n_jobs)
        model = AdaBoostClassifier(random_state=13)
        grid = GridSearchCV(estimator=model,
                            param_grid=model_params,
                            cv=cross_validation,
                            n_jobs=grid_n_jobs,
                            scoring='neg_mean_absolute_error')
        grid.fit(self.__X_train, self.__Y_train.values.ravel())
        self.__grid_best_params = grid.best_params_
        self.__is_grid_fit = True

    def predict(self, data: pd.DataFrame):
        """
        This method predicting values on data
        :param data:
        """
        if not self.__is_grid_fit:
            raise Exception('At first you need to learn model!')
        return self.model.predict(data)

    def get_text_name(self) -> str:
        """
        """
        return self.__text_name

    def get_grid_locked_params(self) -> dict:
        """
        """
        if not self.__is_grid_fit:
            raise Exception('At first you need to learn grid')
        locked = {}
        for param in self.__grid_best_params:
            if param in self.get_locked_params_names():
                locked[param] = self.__grid_best_params[param]
        return locked

    def get_locked_params_names(self) -> List[str]:
        """
        """
        return [p for p in self.__default if self.__default[p].is_locked]

    def get_non_locked_params_names(self) -> List[str]:
        """
        """
        return [p for p in self.__default if not self.__default[p].is_locked]

    def get_default_param_types(self) -> dict:
        """
        """
        default_param_types = {}
        for default in self.__default:
            default_param_types[default] = self.__default[default].ptype
        return default_param_types

    def get_default_param_values(self) -> dict:
        """
        """
        default_param_values = {}
        for default in self.__default:
            default_param_values[default] = self.__default[default].def_val
        return default_param_values

    def get_default_grid_param_values(self) -> dict:
        """
        """
        default_param_values = {}
        for default in self.__default:
            default_param_values[default] = self.__default[default].def_vals
        return default_param_values

    def get_is_model_fit(self) -> bool:
        """
        This method return flag is_model_fit
        """
        return self.__is_model_fit

    def get_is_grid_fit(self) -> bool:
        """
        This method return flag get_is_grid_fit
        """
        return self.__is_grid_fit

    def get_is_dataset_set(self) -> bool:
        """
        This method return flag is_dataset_set
        """
        return self.__is_dataset_set

    def get_grid_best_params(self) -> dict:
        """
        This method return the dict of best params for this model
        """
        if self.__is_grid_fit:
            return self.__grid_best_params
        else:
            raise Exception('At first you need to learn grid')

    def get_feature_importance(self) -> dict:
        """
        This method return dict of feature importance where key is the column of input dataset, and value is importance
        of this column
        """
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        for index in range(len(self.model.feature_importances_)):
            self.__importance[self.__keys[index]] = self.model.feature_importances_[index]
        return {k: v for k, v in sorted(self.__importance.items(), key=lambda item: item[1], reverse=True)}

    def copy(self):
        """
        This method return copy of this class
        """
        return copy.copy(self)

    def get_roc_auc_score(self) -> float:
        """
        This method calculates the "ROC AUC score" for the {self.__text_name} on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_roc_auc_score(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"ROC AUC score\" error!")
        return error

    def get_r_squared_error(self) -> float:
        """
        This method calculates the "R-Squared_error" for the on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_r_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"R-Squared_error\" error!")
        return error

    def get_mean_absolute_error(self) -> float:
        """
        This method calculates the "mean_absolute_error" for the {self.text_name} on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_absolute_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"Mean Absolute Error\" error!")
        return error

    def get_mean_squared_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_mean_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"Mean Squared Error\" error!")
        return error

    def get_root_mean_squared_error(self) -> float:
        """
        This method calculates the "root_mean_squared_error" for the {self.text_name} on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_root_mean_squared_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"Root Mean Squared Error\" error!")
        return error

    def get_median_absolute_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        """
        error = float("inf")
        if not self.__is_model_fit:
            raise Exception(f"You haven't trained the {self.__text_name} yet!")
        try:
            error = Errors.get_median_absolute_error(self.__y_test, self.model.predict(self.__x_test))
        except:
            warnings.warn("An error occurred when calculating the \"Median Absolute Error\" error!")
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

    def set_train_test(self, X_train, x_test, Y_train, y_test) -> None:
        """
        This method sets parameters for the training grid bypassing "set_params". Use it exclusively for the blitz test!
        :param X_train: Training data sample
        :param Y_train: Training value sample
        :param x_test: Test data sample
        :param y_test: Test value sample
        """
        self.__X_train, self.__x_test, self.__Y_train, self.__y_test = X_train, x_test, Y_train, y_test
        self.__is_dataset_set = True

    def __get_default_model_fit_time(self) -> float:
        """
        This method return time of fit model with defualt params
        """
        time_start = time.time()
        model = AdaBoostClassifier(random_state=13)
        model.fit(self.__X_train, self.__Y_train)
        time_end = time.time()
        return time_end - time_start
