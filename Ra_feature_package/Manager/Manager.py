import copy
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from Ra_feature_package.models.Regression.LinRegression import *
from Ra_feature_package.models.Regression.DTRegressor import *
from Ra_feature_package.models.Regression.RFRegressor import *
from Ra_feature_package.models.Regression.ETRegressor import *
from Ra_feature_package.models.Regression.GBRegressor import *
from Ra_feature_package.models.Regression.SGDRegressor import *
from Ra_feature_package.models.Regression.LassoRegressor import *
from Ra_feature_package.models.Regression.LassoCVRegressor import *
from Ra_feature_package.models.Regression.RidgeRegressor import *
from Ra_feature_package.models.Regression.RidgeCVRegressor import *
from Ra_feature_package.models.Regression.BayesRidgeRegressor import *
from Ra_feature_package.models.Regression.ElasticNetRegressor import *
from Ra_feature_package.models.Regression.ElasticNetCVRegressor import *
from Ra_feature_package.models.Regression.LarsRegressor import *
from Ra_feature_package.models.Regression.LarsCVRegressor import *
from Ra_feature_package.models.Regression.HuberRegressor import *
from Ra_feature_package.models.Regression.KNeigRegressor import *
from Ra_feature_package.models.Regression.SVRegressor import *
from Ra_feature_package.models.Regression.LinSVRegressor import *
from Ra_feature_package.models.Regression.ABoostRegressor import *
from Ra_feature_package.models.Regression.BagRegressor import *

# All regressions
# [<class 'sklearn.ensemble.forest.RandomForestRegressor'>,
# <class 'sklearn.ensemble.forest.ExtraTreesRegressor'>,
# <class 'sklearn.ensemble.bagging.BaggingRegressor'>,
# <class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>,
# <class 'sklearn.ensemble.weight_boosting.AdaBoostRegressor'>,
# <class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>,
# <class 'sklearn.isotonic.IsotonicRegression'>,
# <class 'sklearn.linear_model.bayes.ARDRegression'>,
# <class 'sklearn.linear_model.huber.HuberRegressor'>,
# <class 'sklearn.linear_model.base.LinearRegression'>,
# <class 'sklearn.linear_model.logistic.LogisticRegression'>,
# <class 'sklearn.linear_model.logistic.LogisticRegressionCV'>,
# <class 'sklearn.linear_model.passive_aggressive.PassiveAggressiveRegressor'>,
# <class 'sklearn.linear_model.randomized_l1.RandomizedLogisticRegression'>,
# <class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>, <class
# 'sklearn.linear_model.theil_sen.TheilSenRegressor'>, <class
# 'sklearn.linear_model.ransac.RANSACRegressor'>, <class
# 'sklearn.multioutput.MultiOutputRegressor'>, <class
# 'sklearn.neighbors.regression.KNeighborsRegressor'>, <class
# 'sklearn.neighbors.regression.RadiusNeighborsRegressor'>, <class
# 'sklearn.neural_network.multilayer_perceptron.MLPRegressor'>, <class
# 'sklearn.tree.tree.DecisionTreeRegressor'>, <class
# 'sklearn.tree.tree.ExtraTreeRegressor'>, <class
# 'sklearn.svm.classes.SVR'>]


class Regression:
    def __init__(self,
                 task: pd.DataFrame,
                 target: pd.DataFrame,
                 train_split: int):
        self.task = task
        self.target = target
        self.train_split = train_split

        self.models = {"LinearRegression": LinRegressor(),
                       "DecisionTreeRegressor": DTRegressor(),
                       "RandomForestRegressor": RFRegressor(),
                       "ExtraTreesRegressor": ETRegressor(),
                       "GradientBoostingRegressor": GBRegressor(),
                       "StochasticGradientDescentRegressor": SGDRegressor(),
                       "LassoRegressor": LassoRegressor(),
                       "LassoCVRegressor": LassoCVRegressor(),
                       "RidgeRegressor": RidgeRegressor(),
                       "RidgeCVRegressor": RidgeCVRegressor(),
                       "BayesianRidgeRegressor": BayesianRidgeRegressor(),
                       "ElasticNetRegressor": ENetRegressor(),
                       "ElasticNetCVRegressor": ENetCVRegressor(),
                       "LarsRegressor": LarsRegressor(),
                       "LarsCVRegressor": LarsCVRegressor(),
                       "HuberRegressor": HuberRRegressor(),
                       "KNeighborsRegressor": KNRegressor(),
                       "SVRegressor": SVRegressor(),
                       "LinSVRegressor": LinSVRegressor(),
                       "ABoostRegressor": ABoostRegressor()}

    def blitz_test(self,
                   N: int = 3,
                   show: bool = False,
                   prefit: bool = False,
                   n_jobs: int = 1):
        results = {}
        task = self.task
        target = self.target
        X_train, x_test, Y_train, y_test = train_test_split(task,
                                                            target,
                                                            train_size=self.train_split,
                                                            random_state=13)
        simple_test = self.__blitz_test(X_train=X_train,
                                        x_test=x_test,
                                        Y_train=Y_train,
                                        y_test=y_test,
                                        suffix="Simple",
                                        prefit=prefit,
                                        n_jobs=n_jobs)

        SS = StandardScaler()
        SS_task = SS.fit_transform(self.task)

        SS_X_train, SS_x_test, SS_Y_train, SS_y_test = train_test_split(SS_task,
                                                                        self.target,
                                                                        train_size=self.train_split,
                                                                        random_state=13)
        SS_test = self.__blitz_test(X_train=SS_X_train,
                                    x_test=SS_x_test,
                                    Y_train=SS_Y_train,
                                    y_test=SS_y_test,
                                    suffix="SS",
                                    prefit=prefit,
                                    n_jobs=n_jobs)

        MM = StandardScaler()
        MM_task = MM.fit_transform(self.task)

        MM_X_train, MM_x_test, MM_Y_train, MM_y_test = train_test_split(MM_task,
                                                                        self.target,
                                                                        train_size=self.train_split,
                                                                        random_state=13)
        MM_test = self.__blitz_test(X_train=MM_X_train,
                                    x_test=MM_x_test,
                                    Y_train=MM_Y_train,
                                    y_test=MM_y_test,
                                    suffix="MM",
                                    prefit=prefit,
                                    n_jobs=n_jobs)

        N = Normalizer()
        N_task = N.fit_transform(self.task)
        N_X_train, N_x_test, N_Y_train, N_y_test = train_test_split(N_task,
                                                                    self.target,
                                                                    train_size=self.train_split,
                                                                    random_state=13)
        N_test = self.__blitz_test(X_train=N_X_train,
                                   x_test=N_x_test,
                                   Y_train=N_Y_train,
                                   y_test=N_y_test,
                                   suffix="N",
                                   prefit=prefit,
                                   n_jobs=n_jobs)

        results = {**results, **simple_test, **SS_test, **MM_test, **N_test}
        results = dict(sorted(results.items(), key=lambda x: x[1]))
        if show:
            table = PrettyTable()
            table.title = f"Regression model results"
            table.field_names = ["Model", "ROC AUC Score", "R-Squared Error", "Mean Absolute Error",
                                 "Mean Squared Error", "Root Mean Squared Error", "Median Absolute Error"]
            for result in results:
                table.add_row([result] + results[result])
            print(table)

    def __blitz_test(self, X_train, x_test, Y_train, y_test, suffix: str, prefit: bool = False, n_jobs: int = 1):
        results = {}
        for model in self.models:
            try:
                this_model = self.models[model].copy()
                this_model.set_params(task=self.task, target=self.target, train_split=1)
                this_model.set_train_test(X_train=X_train, x_test=x_test,
                                          Y_train=Y_train, y_test=y_test)
                if prefit:
                    this_model.fit_grid(count=0, grid_n_jobs=n_jobs)
                    this_model.fit(grid_params=True, n_jobs=n_jobs)
                else:
                    this_model.fit(n_jobs=n_jobs)
                results[f"{model}_{suffix}"] = [this_model.get_roc_auc_score(),
                                           this_model.get_r_squared_error(),
                                           this_model.get_mean_absolute_error(),
                                           this_model.get_mean_squared_error(),
                                           this_model.get_root_mean_squared_error(),
                                           this_model.get_median_absolute_error()]

            except:
                results[f"{model}_{suffix}"] = [float("inf"), float("inf"), float("inf"),
                                                float("inf"), float("inf"), float("inf")]
        # Сделай возвращение классов, где будет указано, что за модель, какие настройки и что использовано, \
        # можешь сразу возвращать SS, MM, N 
        return results

