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
                         N: int = 5):
    models_params = {}
    results = []
    for model in models:
        try:
            this_model = models[model].copy()
            this_model.set_params(task=task,
                                  target=target,
                                  train_split=train_split,
                                  show=show)
            # this_model.fit_grid(count=0, grid_n_jobs=-1)
            # models_params[model] = this_model.get_grid_best_params()
            # this_model.fit(param_dict=models_params[model], n_jobs=1)
            this_model.fit(n_jobs=-1)
            results.append([model,
                            this_model.get_roc_auc_score(),
                            this_model.get_r_squared_error(),
                            this_model.get_mean_absolute_error(),
                            this_model.get_mean_squared_error(),
                            this_model.get_median_absolute_error()])
        except:
            results.append([model, float("inf"), float("inf"), float("inf"), float("inf"), float("inf")])
    results.sort(key=itemgetter(1, 2, 3, 4, 5), reverse=False)
    if True:
        table = PrettyTable()
        table.title = f"Regression model results"
        table.field_names = ["Model", "ROC AUC Score", "R-Squared Error",
                             "Mean Absolute Error", "Mean Squared Error", "Median Absolute Error"]
        for result in results:
            table.add_row(result)
        print(table)


