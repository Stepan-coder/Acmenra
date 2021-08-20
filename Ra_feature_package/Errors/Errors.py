from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error


def get_roc_auc_score(values1: list, values2: list) -> float:
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return roc_auc_score(values1, values2)


def get_r_squared_error(values1: list, values2: list) -> float:
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return 1.0 - r2_score(values1, values2)


def get_mean_absolute_error(values1: list, values2: list) -> float:
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return mean_absolute_error(values1, values2)


def get_mean_squared_error(values1: list, values2: list) -> float:
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return mean_squared_error(values1, values2)


def get_median_absolute_error(values1: list, values2: list) -> float:
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return median_absolute_error(values1, values2)