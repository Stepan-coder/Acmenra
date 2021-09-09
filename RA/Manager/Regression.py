import copy
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from RA.Manager.ModelBlitzTestResult import *
from RA.models.Regression.LinRegression import *
from RA.models.Regression.DTRegressor import *
from RA.models.Regression.RFRegressor import *
from RA.models.Regression.ETRegressor import *
from RA.models.Regression.PasAggrRegressor import *
from RA.models.Regression.GBRegressor import *
from RA.models.Regression.SGDRegressor import *
from RA.models.Regression.LassoRegressor import *
from RA.models.Regression.LassoCVRegressor import *
from RA.models.Regression.RidgeRegressor import *
from RA.models.Regression.RidgeCVRegressor import *
from RA.models.Regression.BayesRidgeRegressor import *
from RA.models.Regression.ElasticNetRegressor import *
from RA.models.Regression.ElasticNetCVRegressor import *
from RA.models.Regression.LarsRegressor import *
from RA.models.Regression.LarsCVRegressor import *
from RA.models.Regression.HuberRegressor import *
from RA.models.Regression.KNeigRegressor import *
from RA.models.Regression.SVRegressor import *
from RA.models.Regression.LinSVRegressor import *
from RA.models.Regression.GaussPRegressor import *
from RA.models.Regression.IsoRegression import *
from RA.models.Regression.ARDRegression import *
from RA.models.Regression.TSenRegressor import *
from RA.models.Regression.ABoostRegressor import *
from RA.models.Regression.BagRegressor import *
from RA.models.Regression.MLPRegressor import *
from RA.models.Regression.RadNeigRegressor import *
from RA.models.Regression.RANSACRegressor import *


# All regressions
# <class 'sklearn.linear_model.ransac.RANSACRegressor'>


class Regression:
    def __init__(self):
        self.__models = {"AdaBoostRegressor": ABoostRegressor(),
                         "ARDRegression": ARDRegression(),
                         "BaggingRegressor": BagRegressor(),

                         "BayesRidgeRegressor": BayesRidgeRegressor(),
                         "DecisionTreeRegressor": DTRegressor(),
                         "ElasticNetCVRegressor": ENetCVRegressor(),

                         "ElasticNetRegressor": ENetRegressor(),
                         "ExtraTreesRegressor": ETRegressor(),
                         "GaussianProcessRegressor": GaussPRegressor(),

                         "GradientBoostingRegressor": GBRegressor(),
                         "HuberRegressor": HuberRRegressor(),
                         "IsoRegression": IsoRegression(),

                         "KNeighborsRegressor": KNRegressor(),
                         "LarsCVRegressor": LarsCVRegressor(),
                         "LarsRegressor": LarsRegressor(),

                         "LassoCVRegressor": LassoCVRegressor(),
                         "LassoRegressor": LassoRegressor(),
                         "LinearRegression": LinRegressor(),

                         "LinSVRegressor": LinSVRegressor(),
                         "MultiLayerPerceptronRegressor": MLPRegressor(),
                         "PassiveAggressiveRegressor": PasAggrRegressor(),

                         "RadiusNeighborsRegressor": RadNeigRegressor(),
                         "RANSACRegressor": RANSACRegressor(),
                         "RandomForestRegressor": RFRegressor(),
                         "RidgeCVRegressor": RidgeCVRegressor(),

                         "RidgeRegressor": RidgeRegressor(),
                         "StochasticGradientDescentRegressor": SGDRegressor(),
                         "SVRegressor": SVRegressor(),
                         "TSenRegressor": TSenRegressor()}

    def get_model(self, model_name: str):
        if model_name not in self.__models:
            raise Exception("There is no such model!")
        return self.__models[model_name]

    def get_models_names(self) -> List[str]:
        """
        This method return the list of models names
        :return: list of models names
        """
        return list(self.__models.keys())

    def blitz_test(self,
                   task: pd.DataFrame,
                   target: pd.DataFrame,
                   train_split: int,
                   prefit: bool = False,
                   n_jobs: int = 1,
                   show: bool = False) -> list:
        """
        :param task:
        :param target:
        :param train_split:
        :param prefit:
        :param n_jobs: The number of jobs to run in parallel.
        :param show: The parameter responsible for displaying the progress of work
        :return: List of ModelBlitzTestResult classes
        """
        standard_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        max_abs_scaler = MaxAbsScaler()
        normalizer = Normalizer()
        SS_task = standard_scaler.fit_transform(task)
        MMS_task = min_max_scaler.fit_transform(task)
        RS_task = robust_scaler.fit_transform(task)
        MAS_task = max_abs_scaler.fit_transform(task)
        N_task = normalizer.fit_transform(task)
        X_train, x_test, Y_train, y_test = train_test_split(task, target,
                                                            train_size=train_split, random_state=13)
        SS_X_train, SS_x_test, SS_Y_train, SS_y_test = train_test_split(SS_task, target,
                                                                        train_size=train_split, random_state=13)
        MMS_X_train, MMS_x_test, MMS_Y_train, MMS_y_test = train_test_split(MMS_task, target,
                                                                            train_size=train_split, random_state=13)
        RS_X_train, RS_x_test, RS_Y_train, RS_y_test = train_test_split(RS_task, target,
                                                                        train_size=train_split, random_state=13)
        MAS_X_train, MAS_x_test, MAS_Y_train, MAS_y_test = train_test_split(MAS_task, target,
                                                                            train_size=train_split, random_state=13)
        N_X_train, N_x_test, N_Y_train, N_y_test = train_test_split(N_task, target,
                                                                    train_size=train_split, random_state=13)
        simple_test = self.__blitz_test(X_train=X_train, x_test=x_test,
                                        Y_train=Y_train, y_test=y_test,
                                        task=task, target=target,
                                        converter="Simple", prefit=prefit, n_jobs=n_jobs)
        SS_test = self.__blitz_test(X_train=SS_X_train, x_test=SS_x_test,
                                    Y_train=SS_Y_train, y_test=SS_y_test,
                                    task=task, target=target,
                                    converter="StandardScaler", prefit=prefit, n_jobs=n_jobs)
        MMS_test = self.__blitz_test(X_train=MMS_X_train, x_test=MMS_x_test,
                                     Y_train=MMS_Y_train, y_test=MMS_y_test,
                                     task=task, target=target,
                                     converter="MinMaxScaler", prefit=prefit, n_jobs=n_jobs)
        RS_test = self.__blitz_test(X_train=RS_X_train, x_test=RS_x_test,
                                    Y_train=RS_Y_train, y_test=RS_y_test,
                                    task=task, target=target,
                                    converter="RobustScaler", prefit=prefit, n_jobs=n_jobs)
        MAS_test = self.__blitz_test(X_train=MAS_X_train, x_test=MAS_x_test,
                                     Y_train=MAS_Y_train, y_test=MAS_y_test,
                                     task=task, target=target,
                                     converter="MaxAbsScaler", prefit=prefit, n_jobs=n_jobs)
        N_test = self.__blitz_test(X_train=N_X_train, x_test=N_x_test,
                                   Y_train=N_Y_train, y_test=N_y_test,
                                   task=task, target=target,
                                   converter="Normalizer", prefit=prefit, n_jobs=n_jobs)
        results = simple_test + SS_test + MMS_test + RS_test + MAS_test + N_test
        results.sort(key=itemgetter(2, 3, 4, 5, 6, 7),
                     reverse=False)
        if show:
            table = PrettyTable()
            table.title = f"Regression results X {len(results)} models"
            table.field_names = ["Model", "Converter",
                                 "ROC AUC Score", "R-Squared Error", "Mean Absolute Error",
                                 "Mean Squared Error", "Root Mean Squared Error", "Median Absolute Error"]
            for result in results:
                table.add_row(result)
            print(table)
        blitz_results = []
        for result in results:
            blitz_results.append(ModelBlitzTestResult(result))
        return blitz_results

    def __blitz_test(self, X_train, x_test, Y_train, y_test, task: pd.DataFrame, target: pd.DataFrame,
                     converter: str, prefit: bool = False, n_jobs: int = 1):
        results = []
        for model in self.__models:
            try:
                this_model = self.__models[model].copy()
                # this_model.set_params(task=task, target=target, train_split=int(len(task) * 0.5))
                this_model.set_train_test(X_train=X_train, x_test=x_test,
                                          Y_train=Y_train, y_test=y_test)
                if prefit:
                    this_model.fit_grid(count=0, grid_n_jobs=n_jobs)
                    this_model.fit(grid_params=True, n_jobs=n_jobs)
                else:
                    this_model.fit(n_jobs=n_jobs)
                results.append([model, converter,
                                this_model.get_roc_auc_score(), this_model.get_r_squared_error(),
                                this_model.get_mean_absolute_error(), this_model.get_mean_squared_error(),
                                this_model.get_root_mean_squared_error(), this_model.get_median_absolute_error()])
            except:
                results.append([model, converter,
                                float("inf"), float("inf"), float("inf"),
                                float("inf"), float("inf"), float("inf")])
        return results