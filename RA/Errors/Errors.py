import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error


def get_roc_auc_score(values1: list, values2: list) -> float:
    """
    Calculate the ROC AUC score between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The ROC AUC score between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return roc_auc_score(values1, values2)


def get_r_squared_error(values1: list, values2: list) -> float:
    """
    Calculate the R-squared error between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The R-squared error between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return 1.0 - r2_score(values1, values2)


def get_mean_absolute_error(values1: list, values2: list) -> float:
    """
    Calculate the mean absolute error between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The mean absolute error between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return mean_absolute_error(values1, values2)


def get_mean_squared_error(values1: list, values2: list) -> float:
    """
    Calculate the mean squared error between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The mean squared error between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return mean_squared_error(values1, values2)


def get_root_mean_squared_error(values1: list, values2: list) -> float:
    """
    Calculate the root mean squared error between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The root mean squared error between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return math.sqrt(mean_squared_error(values1, values2))


def get_median_absolute_error(values1: list, values2: list) -> float:
    """
    Calculate the median absolute error between two lists of values.

    Args:
        values1 (list): The first list of values.
        values2 (list): The second list of values.

    Returns:
        float: The median absolute error between the two lists.
    """
    if len(values1) != len(values2):
        raise Exception(f"The presented lists have different lengths: values1: {len(values1)}, values2: {len(values2)}")
    return median_absolute_error(values1, values2)
