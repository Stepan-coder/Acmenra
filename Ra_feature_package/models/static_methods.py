import math


def show_grid_params(params: dict, locked_params: list, single_model_time, n_jobs: int):
    """
    This method show grid parameters from dict 'params'
    :param params: Dict of grid params
    """
    count_elements = []
    multiply = 1
    for param in params:
        param_text = "  -Param \'{0}\'({2}): {1}".format(param, params[param], len(params[param]))
        if param in locked_params:
            print("\033[31m {}".format(param_text))
        else:
            print("\033[32m {}".format(param_text))
        count_elements.append(len(params[param]))

    for ce in count_elements:
        multiply *= ce
    total_models = "Total {1} models: {0}".format(" X ".join([str(e) for e in count_elements]), str(multiply))
    fit_time = "~" + convert_to_preferred_format(multiply * single_model_time / n_jobs)
    total_time = "Total time: {0}, {1} per model * {2} flows.".format(fit_time, round(single_model_time, 2), n_jobs)
    print("\033[0m {}".format(total_models))
    print("\033[0m {}".format(total_time))


def conf_params(min_val: int or float,
                count: int or float,
                ltype: type,
                max_val: int or float = None) -> list:
    if max_val is not None:
        step = (float(max_val) - float(min_val)) / float(count)
    else:
        step = 1
    if ltype == int:
        return list(set([int(min_val + val * step) for val in range(count)]))
    else:
        return list(set([float(min_val + val * step) for val in range(count)]))


def get_choosed_params(params: list, count: int, ltype: type) -> list:
    """
    This method calculates the values with the specified step
    :param params: The list of input parameters
    :param count: The step with which to return the values
    :return: The step with which to return the values
    """
    if len(params) < 1:
        raise Exception("The list of \'params\' should not be empty!")
    if count < 1:
        raise Exception("The value of \'count\' must be greater than 0!")
    if count > len(params):
        count = len(params)
    if len(params) == 1:
        return params

    has_none = False
    if None in params:
        params = [value for value in params if value is not None]
        count -= 1
        has_none = True

    if len(params) == 0:
        return [None]
    elif count == 1:
        remains_params = [params[int(len(params) / 2)]]
    elif count == 2:
        remains_params = [params[int(1 * len(params) / 3)], params[int(2 * len(params) / 3)]]
    elif count == 3:
        remains_params = [params[0], params[int(len(params) / 2)], params[-1]]
    else:
        params.sort()
        index = float(len(params)) / float(((count + 1) - 2))
        remains_params = [params[0], params[-1]]
        for i in range(((count + 1) - 2)):
            if ltype == int:
                remains_params.append(int(math.ceil(params[i] * index)))
            else:
                remains_params.append(params[i] * index)
    remains_params = list(set(remains_params))
    remains_params.sort()
    if has_none:
        remains_params.append(None)
    return remains_params


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
        check_params_list(grid_param, value, param_type)
    else:
        check_param_value(grid_param, value, param_type)


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


def convert_to_preferred_format(sec: float) -> str:
    """
    This method conver seconds to format HH:MM:SS
    :param sec: Value in seconds
    :return:
    """
    sec = sec % (24 * 3600)
    hour = sec / 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)
