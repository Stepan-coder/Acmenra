import copy

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
from Ra_feature_package.models.Regression.ElasticNetCVRegressor import *
from Ra_feature_package.models.Regression.ElasticNetRegressor import *
from Ra_feature_package.models.Regression.LarsRegressor import *
from Ra_feature_package.models.Regression.LarsCVRegressor import *
from Ra_feature_package.models.Regression.HuberRegressor import *
from Ra_feature_package.models.Regression.BayesRidgeRegressor import *
from Ra_feature_package.models.Regression.ABoostRegressor import *
from Ra_feature_package.models.Regression.BagRegressor import *
from Ra_feature_package.models.Regression.KNeigRegressor import *
from Ra_feature_package.models.Regression.SVRegressor import *
from Ra_feature_package.models.Regression.LinSVRegressor import *

# Кароче, пропиши везде защиты (проверка на то, что модель/сетка обучены) \
# и сделай везде сет перамс
# Ну и добавь метод self.copy() для каждого класса, чтобы происходило именно копирование...


#  Нужен блитц тест моделей
#  Надо сделать так, чтобы значения об ошибках усреднялись (заметил прикол, \
#  что если перезапускать модель, резы отличаются


def blitz_test_regressor(task: pd.DataFrame,
                         target: pd.DataFrame,
                         train_split: int,
                         show: bool = False):
    l = LinRegressor(task=task,
                     target=target,
                     train_split=train_split,
                     show=show)
    cl = copy.copy(l)
    l.fit()
    print(l)
    print(cl)

