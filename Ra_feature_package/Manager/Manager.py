import copy
from operator import itemgetter

from Ra_feature_package.models.Classifier.LogRegression import *
from Ra_feature_package.models.Classifier.DTClassifier import *
from Ra_feature_package.models.Classifier.RFClassifier import *


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

# Кароче, пропиши везде защиты (проверка на то, что модель/сетка обучены) \
# и сделай везде сет перамс
# Ну и добавь метод self.copy() для каждого класса, чтобы происходило именно копирование...


#  Нужен блитц тест моделей
#  Надо сделать так, чтобы значения об ошибках усреднялись (заметил прикол, \
#  что если перезапускать модель, резы отличаются
models = {"LinearRegression": LinRegressor(),
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


def blitz_test_regressor(task: pd.DataFrame,
                         target: pd.DataFrame,
                         train_split: int,
                         show: bool = False,
                         prefit: bool = False,
                         n_jobs: int = 1):
    models_params = {}
    results = []
    X_train, x_test, Y_train, y_test = train_test_split(task, target, train_size=train_split, random_state=13)
    for model in models:
        try:
            this_model = models[model].copy()
            this_model.set_params(task=task, target=target, train_split=train_split, show=False)
            this_model.set_train_test(X_train=X_train, x_test=x_test,
                                      Y_train=Y_train, y_test=y_test)
            if prefit:
                this_model.fit_grid(count=0, grid_n_jobs=n_jobs)
                models_params[model] = this_model.get_grid_best_params()
                this_model.fit(grid_params=True, n_jobs=n_jobs)
            else:
                models_params[model] = this_model.get_default_param_values()
                this_model.fit(n_jobs=n_jobs)
            results.append([model,
                            this_model.get_roc_auc_score(),
                            this_model.get_r_squared_error(),
                            this_model.get_mean_absolute_error(),
                            this_model.get_mean_squared_error(),
                            this_model.get_median_absolute_error()])

        except:
            results.append([model, float("inf"), float("inf"), float("inf"), float("inf"), float("inf")])
    results.sort(key=itemgetter(1, 2, 3, 4, 5), reverse=False)
    if show:
        table = PrettyTable()
        table.title = f"Regression model results"
        table.field_names = ["Model", "ROC AUC Score", "R-Squared Error",
                             "Mean Absolute Error", "Mean Squared Error", "Median Absolute Error"]
        for result in results:
            table.add_row(result)
        print(table)
    return results[0][0], models_params[results[0][0]]


