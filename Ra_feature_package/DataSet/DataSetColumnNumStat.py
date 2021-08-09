import math
from typing import Dict, List


class NormalDistribution:
    def __init__(self, values: List[int or float] = None):
        self.__math_mode = None
        self.__math_expectation = None
        self.__math_dispersion = None
        self.__math_sigma = None
        self.__math_distribution = None
        self.__is_normal_distribution = False
        if values is not None:
            self.__fill_normal_distribution(values=values)

    def get_math_distribution(self) -> Dict[int or float, float]:
        """
        This method return mathematical distribution dict
        """
        return self.__math_distribution

    def get_math_mode(self):
        """
        This method return mathematical mode
        """
        return self.__math_mode

    def get_math_expectation(self):
        """
        This method return mathematical expectation
        """
        return self.__math_expectation

    def get_math_dispersion(self):
        """
        This method return mathematical dispersion
        """
        return self.__math_dispersion

    def get_math_sigma(self):
        """
        This method return mathematical sigma
        """
        return self.__math_sigma

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
        return {"Math mode": self.__math_mode,
                "Math expectation": self.__math_expectation,
                "Math dispersion": self.__math_dispersion,
                "Math sigma": self.__math_sigma,
                "Moda - 3 * sigma": self.__math_expectation - 3 * self.__math_sigma,
                "Moda - 2 * sigma": self.__math_expectation - 2 * self.__math_sigma,
                "Moda - 1 * sigma": self.__math_expectation - 1 * self.__math_sigma,
                "Moda + 1 * sigma": self.__math_expectation + 1 * self.__math_sigma,
                "Moda + 2 * sigma": self.__math_expectation + 2 * self.__math_sigma,
                "Moda + 3 * sigma": self.__math_expectation + 3 * self.__math_sigma}

    def set_values(self, values: List[int or float]):
        if not self.__is_normal_distribution:
            if values is not None:
                self.__fill_normal_distribution(values=values)

    def __fill_normal_distribution(self, values: List[int or float]) -> None:
        """
        This method fill NormalDistribution class when we use values
        :param values: list of column values
        """

        self.__math_distribution = self.__get_math_distribution(values)
        self.__math_mode = self.__get_math_moda(self.__math_distribution)
        self.__math_expectation = self.__get_math_expectation(self.__math_distribution)
        self.__math_dispersion = self.__get_math_dispersion(self.__math_distribution)
        if self.__math_dispersion >= 0:
            self.__math_sigma = math.sqrt(self.__math_dispersion)
        else:
            self.__math_sigma = -1 * math.sqrt(abs(self.__math_dispersion))
        self.__is_normal_distribution = True

    @staticmethod
    def __get_math_distribution(values: list) -> Dict[int or float, float]:
        """
        This method calculates the frequency of list values as a percentage
        :return: dict
        """
        math_rasp = {}
        for value in values:
            if value in list(math_rasp):
                math_rasp[value] += (1 / len(values))
            else:
                math_rasp[value] = (1 / len(values))
        return math_rasp

    @staticmethod
    def __get_math_dispersion(math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical variance
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait = NormalDistribution.__get_math_expectation(math_rasp_dict)
        math_wait_x2 = math_wait * math_wait
        math_wait_2 = NormalDistribution.__get_math_wait_2(math_rasp_dict)
        return math_wait_2 - math_wait_x2

    @staticmethod
    def __get_math_moda(math_rasp_dict: dict):
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
    def __get_math_expectation(math_rasp_dict: dict) -> float:
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
    def __get_math_wait_2(math_rasp_dict: dict) -> float:
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
    def __init__(self, values: List[int or float] = None, normal_distribution: bool = False):
        self.__min = None
        self.__max = None
        self.__average = None
        self.normal_distribution = None
        self.__use_normal_distribution = normal_distribution
        self.__is_numerical_indicators = False  # Отвечает за наличие данных в классе NumericalIndicators
        self.__is_normal_distribution = False  # Отвечает за наличие данных в классе NormalDistribution
        if values is not None:
            self.__fill_numerical_indicators(values=values,
                                             normal_distribution=normal_distribution)

    def get_min(self):
        """
        This method return minimal value of column
        """
        return self.__min

    def get_max(self):
        """
        This method return maximal value of column
        """
        return self.__max

    def get_average(self) -> float:
        """
        This method return maximal value of column
        """
        return self.__average

    def get_is_normal_distribution(self) -> bool:
        return self.__is_normal_distribution

    def get_normal_distribution(self) -> NormalDistribution:
        if self.__is_normal_distribution:
            return self.normal_distribution

    def get_is_numerical_indicators(self) -> bool:
        """
        This method return the current state of filling numerical indicators
        """
        return self.__is_numerical_indicators

    def set_values(self, values: List[int or float], normal_distribution: bool = False):
        """
        This method init the filling params of this class
        :param values: Values from DataSetColumn - values of column from the dataset
        :param normal_distribution: The switch responsible for calculating the indicators of the normal distribution
        :return: None
        """
        if not self.__is_numerical_indicators:
            if values is not None:
                self.__fill_numerical_indicators(values=values,
                                                 normal_distribution=normal_distribution)

    def get_from_json(self, data: dict) -> None:
        """
        This method load NumericalIndicators indicators from json
        :param data: Incoming data in json format
        :return: None
        """
        required_fields = ["Minimal value", "Maximal value", "Average value"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__min = data["Minimal value"]
        self.__max = data["Maximal value"]
        self.__average = data["Average value"]
        self.__is_numerical_indicators = True
        if "Normal distribution" in data:
            self.normal_distribution = NormalDistribution()
            self.__is_normal_distribution = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json data
        """
        if not self.__is_numerical_indicators:
            raise Exception("The values were not loaded!")
        data = {"Minimal value": self.__min,
                "Maximal value": self.__max,
                "Average value": self.__average}
        if self.__use_normal_distribution and self.__is_normal_distribution:
            data['Normal distribution'] = self.normal_distribution.to_json()
        return data

    def __fill_numerical_indicators(self, values: List[int or float], normal_distribution: bool = False):
        """
        This method fill this class when we use values
        :param values: list of column values
        :param normal_distribution: The switch responsible for calculating the indicators of the normal distribution
        """
        self.__min = min(values)
        self.__max = max(values)
        self.__average = sum(values) / len(values)
        self.__use_normal_distribution = normal_distribution
        self.__is_numerical_indicators = True
        if normal_distribution and not self.__is_normal_distribution:
            self.normal_distribution = NormalDistribution()
            self.normal_distribution.set_values(values=values)
            self.__is_normal_distribution = True
