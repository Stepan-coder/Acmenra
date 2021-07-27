import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from typing import Dict, List

from Ra_feature_package.Errors import Errors


class LogRegressor:
    def __init__(self,
                 task: pd.DataFrame,
                 target: pd.DataFrame,
                 train_split: int,
                 show: bool = False):
        """
        This method is the initiator of the LogRegressor class
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param show: The parameter responsible for displaying the progress of work
        """
        self.text_name = "LinearRegression"
        self.default_param_types = {'penalty': str,
                                    'dual': bool,
                                    'tol': float,
                                    'C': float,
                                    'fit_intercept': bool,

                                    'intercept_scaling': float,
                                    'class_weight': dict,
                                    'solver': str,
                                    'max_iter': int,
                                    'multi_class': str,

                                    'verbose': int,
                                    'warm_start': bool,
                                    'n_jobs': int,
                                    'l1_ratio': float}

        self.default_param = {'penalty': "l2",
                              'dual': False,
                              'tol': 1e-4,
                              'C': 1.0,
                              'fit_intercept': True,
                              'intercept_scaling': 1,
                              'class_weight': None,
                              'solver': "lbfgs",
                              'max_iter': 100,
                              'multi_class': 'auto',
                              'verbose': 0,
                              'warm_start': False,
                              'n_jobs': None,
                              'l1_ratio': None}

        self.default_params = {'penalty': ["l2", "elasticnet", "none"],
                               'dual': [False],
                               'tol': [1e-4],  # Лучше не трогать
                               'C': [1.0],  # Лучше не трогать
                               'fit_intercept': [True, False],
                               'intercept_scaling': [1],
                               'class_weight': [None],  # Лучше не трогать
                               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                               'max_iter': [i * 10 for i in range(1, len(task.keys()) + 1)],
                               'multi_class': ['auto', 'ovr', 'multinomial'],
                               'verbose': [i for i in range(1, len(task.keys()) + 1)],
                               'warm_start': [True, False],
                               'n_jobs': [i for i in range(1, len(task.keys()) + 1)],
                               'l1_ratio': [None]}

        self.importance = {}
        self.is_model_fit = False
        self.is_grid_fit = False

        self.show = show
        self.model = None
        self.grid_best_params = None
        self.keys = task.keys()
        self.keys_len = len(task.keys())
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(task,
                                                                                target,
                                                                                train_size=train_split,
                                                                                random_state=13)

    def __str__(self):
        return f"'<Ra.{LogRegressor.__name__} model>'"

    def __repr__(self):
        return f"'<Ra.{LogRegressor.__name__} model>'"

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def fit(self,
            param_dict: Dict[str, int or str] = None,
            grid_params: bool = False):
        f"""
        This method trains the model {self.text_name}, it is possible to use the parameters from "fit_grid"
        :param param_dict: The parameter of the hyperparameter grid that we check
        :param grid_params: The switcher which is responsible for the ability to use all the ready-made parameters
         from avia for training
        """
        if grid_params and param_dict is None:
            self.model = LogisticRegression(penalty=self.grid_best_params['penalty'],
                                            dual=self.grid_best_params['dual'],
                                            tol=self.grid_best_params['tol'],
                                            C=self.grid_best_params['C'],
                                            fit_intercept=self.grid_best_params['fit_intercept'],
                                            intercept_scaling=self.grid_best_params['intercept_scaling'],
                                            class_weight=self.grid_best_params['class_weight'],
                                            solver=self.grid_best_params['solver'],
                                            max_iter=self.grid_best_params['max_iter'],
                                            multi_class=self.grid_best_params['multi_class'],
                                            verbose=self.grid_best_params['verbose'],
                                            warm_start=self.grid_best_params['warm_start'],
                                            n_jobs=self.grid_best_params['n_jobs'],
                                            l1_ratio=self.grid_best_params['l1_ratio'],
                                            random_state=13)
        elif not grid_params and param_dict is not None:
            model_params = self.default_param
            for param in param_dict:
                if param not in self.default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                self.check_param(param,
                                 param_dict[param],
                                 self.default_param_types[param],
                                 type(self.default_param[param]))
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
                                            verbose=model_params['verbose'],
                                            warm_start=model_params['warm_start'],
                                            n_jobs=model_params['n_jobs'],
                                            l1_ratio=model_params['l1_ratio'],
                                            random_state=13)
        elif not grid_params and param_dict is None:
            self.model = LogisticRegression()
        else:
            raise Exception("You should only choose one way to select hyperparameters!")
        print(f"Learning {self.text_name}...")
        self.model.fit(self.X_train, self.Y_train.values.astype(float))
        self.is_model_fit = True

    def fit_grid(self,
                 params_dict: Dict[str, list] = None,
                 step: int = 1,
                 cross_validation: int = 3):
        """
        This method uses iteration to find the best hyperparameters for the model and trains the model using them
        :param params_dict: The parameter of the hyperparameter grid that we check
        :param step: The step with which to return the values
        :param cross_validation: The number of sections into which the dataset will be divided for training
        """
        if params_dict is not None:
            for param in params_dict:
                if param not in self.default_params.keys():
                    raise Exception(f"The column {param} does not exist in the set of allowed parameters!")
                self.check_param(grid_param=param,
                                 value=params_dict[param],
                                 param_type=self.default_param_types[param],
                                 setting_param_type=type(self.default_params[param]))
                self.default_params[param] = params_dict[param]

        for param in ['max_iter', 'verbose', 'n_jobs']:
            self.default_params[param] = self.get_choosed_params(self.default_params[param], step=step)

        if self.show:
            print(f"Learning GridSearch {self.text_name}...")
            self.show_grid_params(self.default_params)
        model = LogisticRegression(random_state=13)
        grid = GridSearchCV(model, self.default_params, cv=cross_validation)
        grid.fit(self.X_train, self.Y_train.values.ravel())
        self.grid_best_params = grid.best_params_
        self.is_grid_fit = True

    def get_default_param_types(self) -> dict:
        """
        :return: This method return default model param types
        """
        return self.default_param_types

    def get_default_param_values(self) -> dict:
        """
        :return: This method return default model param values
        """
        return self.default_param

    def get_default_grid_param_values(self) -> dict:
        """
        :return: This method return default model param values for grid search
        """
        return self.default_params

    def get_is_model_fit(self) -> bool:
        f"""
        This method return flag is_model_fit
        :return: is_model_fit
        """
        return self.is_model_fit

    def get_is_grid_fit(self) -> bool:
        f"""
        This method return flag get_is_grid_fit
        :return: get_is_grid_fit
        """
        return self.is_grid_fit

    def get_grid_best_params(self) -> dict:
        """
        This method return the dict of best params for this model
        :return: dict of best params for this model
        """
        if self.is_grid_fit:
            return self.grid_best_params
        else:
            raise Exception('At first you need to learn grid')

    def get_feature_importance(self) -> dict:
        """
        This method return dict of feature importance where key is the column of input dataset, and value is importance
        of this column
        :return: dict of column importance
        """
        if not self.is_model_fit:
            raise Exception(f"You haven't trained the {self.text_name} yet!")
        for index in range(len(self.model.feature_importances_)):
            self.importance[self.keys[index]] = self.model.feature_importances_[index]
        return {k: v for k, v in sorted(self.importance.items(), key=lambda item: item[1], reverse=True)}

    def get_roc_auc_score(self) -> float:
        f"""
        This method calculates the "roc_auc_score" for the {self.text_name} on the test data
        :return: roc_auc_score
        """
        if not self.is_model_fit:
            raise Exception(f"You haven't trained the {self.text_name} yet!")
        return Errors.get_roc_auc_score(self.y_test, self.model.predict(self.x_test))

    def get_mean_squared_error(self) -> float:
        """
        This method calculates the "mean_squared_error" for the {self.text_name} on the test data
        :return: mean_squared_error
        """
        if not self.is_model_fit:
            raise Exception(f"You haven't trained the {self.text_name} yet!")
        return Errors.get_mean_squared_error(self.y_test, self.model.predict(self.x_test))

    def get_mean_absolute_error(self) -> float:
        """
        This method calculates the "mean_absolute_error" for the {self.text_name} on the test data
        :return: mean_absolute_error
        """
        if not self.is_model_fit:
            raise Exception(f"You haven't trained the {self.text_name} yet!")
        return Errors.get_mean_absolute_error(self.y_test, self.model.predict(self.x_test))

    @staticmethod
    def show_grid_params(params: dict):
        """
        This method show grid parameters from dict 'params'
        :param params: Dict of grid params
        """
        count_elements = []
        multiply = 1
        for param in params:
            print("Param({0}[{2}]): {1}".format(param, params[param], len(params[param])))
            count_elements.append(len(params[param]))
        for ce in count_elements:
            multiply *= ce
        print("Total({1}): {0}".format(" X ".join([str(e) for e in count_elements]), multiply))

    @staticmethod
    def get_choosed_params(params: list, step: int) -> list:
        """
        This method calculates the values with the specified step
        :param params: The list of input parameters
        :param step: The step with which to return the values
        :return: The step with which to return the values
        """
        first_param = params[0]
        last_param = params[-1]
        remains_params = params[1:-1]
        choosed_params = remains_params[1::step]
        choosed_params = [first_param] + choosed_params + [last_param]
        choosed_params = list(set(choosed_params))
        choosed_params.sort()
        return choosed_params

    @staticmethod
    def check_param(grid_param: str,
                    value: list or int or str,
                    param_type: type,
                    setting_param_type: type):
        """
        This method switches the check between two methods "_check_params"[for checking values as lists] and
        "_check_param"[for checking values as simplest]
        :param grid_param: The parameter of the hyperparameter grid that we check
        :param value: Values that will be passed to the " grid"
        :param param_type: The data type acceptable for this parameter
        :param setting_param_type: The parameter responsible for selecting the method that will check the input values
        """
        if setting_param_type == list:
            LogRegressor.check_params_list(grid_param, value, param_type)
        else:
            LogRegressor.check_param_value(grid_param, value, param_type)

    @staticmethod
    def check_param_value(grid_param: str,
                          value: str or int,
                          param_type: type):
        """
        This method checks the correctness of the data types passed for training
        :param grid_param: The parameter of the hyperparameter grid that we check
        :param value: Values that will be passed to the " grid"
        :param param_type: The data type acceptable for this parameter
        """
        if not isinstance(value, param_type):
            raise Exception(f"The value of the \'{grid_param}\' parameter must be a \'{param_type}\',"
                            f" byt was \'{type(value)}\'")

    @staticmethod
    def check_params_list(grid_param: str,
                          value: list,
                          param_type: type):
        """
        This method checks the correctness of the data types passed to the " grid"
        :param grid_param: The parameter of the hyperparameter grid that we check
        :param value: Values that will be passed to the " grid"
        :param param_type: The data type acceptable for this parameter
        """
        if isinstance(value, list) and len(value):
            for val in value:
                if not isinstance(val, param_type) and val is not None:
                    raise Exception(f"The value of the \'{grid_param}\' parameter must be a \'{param_type}\',"
                                    f" byt was \'<{type(val)}>\'")
        else:
            raise Exception(f"The value of the '{grid_param}' parameter must be a non-empty list")
