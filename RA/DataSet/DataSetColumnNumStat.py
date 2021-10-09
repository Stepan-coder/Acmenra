import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
from prettytable import PrettyTable


class NormalDistribution:
    def __init__(self):
        self.__math_mode = None
        self.__math_expectation = None
        self.__math_dispersion = None
        self.__math_sigma = None
        self.__math_distribution = None
        self.__coef_of_variation = None
        self.__z_score = None
        self.__is_normal_distribution = False

    def __str__(self):
        table = PrettyTable()
        table.title = f"\"NormalDistribution\""
        table.field_names = ["Indicator", "Value"]
        if self.__is_normal_distribution:
            table.add_row(["Math mode", self.get_math_mode()])
            table.add_row(["Math expectation", self.get_math_expectation()])
            table.add_row(["Math dispersion", self.get_distribution()])
            table.add_row(["Math sigma", self.get_math_sigma()])
            table.add_row(["Moda - 3 * sigma", self.get_math_expectation() - 3 * self.get_math_sigma()])
            table.add_row(["Moda - 2 * sigma", self.get_math_expectation() - 2 * self.get_math_sigma()])
            table.add_row(["Moda - 1 * sigma", self.get_math_expectation() - 1 * self.get_math_sigma()])
            table.add_row(["Moda + 1 * sigma", self.get_math_expectation() + 1 * self.get_math_sigma()])
            table.add_row(["Moda + 2 * sigma", self.get_math_expectation() + 2 * self.get_math_sigma()])
            table.add_row(["Moda + 3 * sigma", self.get_math_expectation() + 3 * self.get_math_sigma()])
        return str(table)

    def get_distribution(self) -> Dict[int or float, float]:
        """
        This method return mathematical distribution dict
        """
        if self.__is_normal_distribution is not None:
            return self.__math_distribution
        else:
            raise Exception("The data has not been loaded yet!")

    def get_math_mode(self) -> int or float:
        """
        This method return mathematical mode
        """
        if self.__is_normal_distribution is not None:
            return self.__math_mode
        else:
            raise Exception("The data has not been loaded yet!")

    def get_math_expectation(self) -> int or float:
        """
        This method return mathematical expectation
        """
        if self.__is_normal_distribution is not None:
            return self.__math_expectation
        else:
            raise Exception("The data has not been loaded yet!")

    def get_math_dispersion(self) -> float:
        """
        This method return mathematical dispersion
        """
        if self.__is_normal_distribution is not None:
            return self.__math_dispersion
        else:
            raise Exception("The data has not been loaded yet!")

    def get_math_sigma(self) -> float:
        """
        This method return mathematical sigma
        """
        if self.__is_normal_distribution is not None:
            return self.__math_sigma
        else:
            raise Exception("The data has not been loaded yet!")

    def get_coef_of_variation(self) -> float:
        """
        This method return Coefficient of Variation
        """
        if self.__is_normal_distribution is not None:
            return self.__coef_of_variation
        else:
            raise Exception("The data has not been loaded yet!")

    def get_Z_score(self) -> float:
        """
        This method return the Z-score
        """
        if self.__is_normal_distribution is not None:
            return self.__z_score
        else:
            raise Exception("The data has not been loaded yet!")

    def get_is_normal_distribution(self) -> bool:
        """
        This method return the current state of filling normal distribution
        """
        return self.__is_normal_distribution

    def from_json(self, data) -> None:
        """
        This method load class fields from json
        :param data: Input json object
        :return: None
        """
        required_fields = ["Math mode", "Math mode", "Math dispersion", "Math sigma"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__math_mode = data["Math mode"]
        self.__math_expectation = data["Math expectation"]
        self.__math_dispersion = data["Math dispersion"]
        self.__math_sigma = data["Math sigma"]
        self.__is_normal_distribution = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json object
        """
        if not self.__is_normal_distribution:
            raise Exception("The values were not loaded!")
        return {"Math mode": float(self.__math_mode),
                "Math expectation": float(self.__math_expectation),
                "Math dispersion": float(self.__math_dispersion),
                "Math sigma": float(self.__math_sigma),
                "Moda - 3 * sigma": float(self.__math_expectation - 3 * self.__math_sigma),
                "Moda - 2 * sigma": float(self.__math_expectation - 2 * self.__math_sigma),
                "Moda - 1 * sigma": float(self.__math_expectation - 1 * self.__math_sigma),
                "Moda + 1 * sigma": float(self.__math_expectation + 1 * self.__math_sigma),
                "Moda + 2 * sigma": float(self.__math_expectation + 2 * self.__math_sigma),
                "Moda + 3 * sigma": float(self.__math_expectation + 3 * self.__math_sigma)}

    def set_values(self, values: List[int or float]) -> None:
        """
        This method calculated extended satatistisc params
        :param values:
        :return: None
        """
        if not self.__is_normal_distribution:
            if values is not None:
                self.__math_distribution = self.__get_values_distribution(values)
                self.__math_mode = stats.mode(values)[0][0]
                self.__math_expectation = self.__get_math_expectation(self.__math_distribution)
                self.__math_sigma = np.std(values)
                self.__math_dispersion = self.__math_sigma ** 2
                self.__coef_of_variation = np.std(values) / np.mean(values) * 100
                self.__z_score = stats.zscore(values)
                self.__is_normal_distribution = True

    @staticmethod
    def __get_values_distribution(values: list) -> Dict[bool or int or float or bool, float]:
        """
        This method calculates the frequency of list values as a percentage
        :return: Dict[bool or int or float or bool, float]
        """
        math_rasp = {}
        for value in values:
            if value in list(math_rasp):
                math_rasp[value] += (1 / len(values))
            else:
                math_rasp[value] = (1 / len(values))
        return math_rasp

    @staticmethod
    def __get_math_dispersion(math_rasp_dict: dict) -> int or float:
        """
        This method calculates the mathematical variance
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: int or float
        """
        math_wait = NormalDistribution.__get_math_expectation(math_rasp_dict)
        math_wait_x2 = math_wait * math_wait
        math_wait_2 = NormalDistribution.__get_math_wait_2(math_rasp_dict)
        return math_wait_2 - math_wait_x2

    @staticmethod
    def __get_math_moda(math_rasp_dict: dict) -> int or float:
        """
        This method calculates the mathematical mode (the most frequent value)
        :param math_rasp_dict: Dictionary of the frequency of values
        :return:
        """
        moda = -10000000000
        moda_key = -1
        for key in math_rasp_dict:
            if math_rasp_dict[key] > moda:
                moda = math_rasp_dict[key]
                moda_key = key
        return moda_key

    @staticmethod
    def __get_math_expectation(math_rasp_dict: dict) -> int or float:
        """
        This method calculates the mathematical expectation
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait = 0
        for md in math_rasp_dict:
            math_wait += float(md) * math_rasp_dict[md]
        return math_wait

    @staticmethod
    def __get_math_wait_2(math_rasp_dict: dict) -> int or float:
        """
        This method calculates the mathematical expectation squared
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait_2 = 0
        for md in math_rasp_dict:
            math_wait_2 += float(md) * float(md) * math_rasp_dict[md]
        return math_wait_2


class NumericalIndicators:
    def __init__(self, values: List[int or float], extended: bool) -> None:
        """
        This method init work of class
        :param values: Values from column
        :param extended: Param switched brtween simple or extended statistics
        :return None
        """
        values = [val for val in values if not math.isnan(val)]
        self.__min = min(values)
        self.__mean = np.mean(values)
        self.__median = np.median(values)
        self.__max = max(values)
        self.__normal_distribution = None
        self.__is_extended = extended
        self.__is_normal_distribution = False
        if extended:
            # It is easier and clearer for us to recalculate basic statistics than to pile up incomprehensible code
            # To download advanced statistics from a json file, you can calculate "basic statistics",
            # because it's not long.
            self.values_distribution = NormalDistribution()
            self.values_distribution.set_values(values=values)
            self.__is_normal_distribution = True  # Отвечает за наличие данных в классе NormalDistribution

    def __str__(self) -> str:
        """
        This method shows the column base info
        :return: str
        """
        table = PrettyTable()
        table.title = f"\"NumericalIndicators\""
        table.field_names = ["Indicator", "Value"]
        table.add_row(["Min val", self.get_min()])
        table.add_row(["Mean val", self.get_mean()])
        table.add_row(["Median val", self.get_median()])
        table.add_row(["Max val", self.get_max()])
        return str(table)

    def get_min(self) -> int or float:
        """
        This method return minimal value of column
        :return int or float
        """
        return self.__min

    def get_max(self) -> int or float:
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        return self.__max

    def get_mean(self) -> int or float:
        """
        This method return mean value of column
        :return Mean value of column
        """
        return self.__mean

    def get_median(self) -> int or float:
        """
        This method return median value of column
        :return Median value of column
        """
        return self.__median

    def get_is_extended(self) -> bool:
        return self.__is_normal_distribution

    def get_values_distribution(self) -> NormalDistribution:
        if self.__is_normal_distribution:
            return self.values_distribution

    def get_from_json(self, data: dict) -> None:
        """
        This method load NumericalIndicators indicators from json
        :param data: Incoming data in json format
        :return: None
        """
        required_fields = ["Minimal value", "Maximal value", "Mean value", "Median value"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__min = data["Minimal value"]
        self.__max = data["Maximal value"]
        self.__mean = data["Mean value"]
        self.__median = data["Median value"]
        self.__is_numerical_indicators = True
        if "Normal distribution" in data:
            self.values_distribution = NormalDistribution()
            self.values_distribution.from_json(data["Normal distribution"])
            self.__is_normal_distribution = True
            self.__use_normal_distribution = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json data
        """
        if not self.__is_numerical_indicators:
            raise Exception("The values were not loaded!")
        data = {"Minimal value": self.__min,
                "Maximal value": self.__max,
                "Mean value": self.__mean,
                "Median value": self.__median}
        if self.__use_normal_distribution and self.__is_normal_distribution:
            data['Normal distribution'] = self.values_distribution.to_json()
        return data

