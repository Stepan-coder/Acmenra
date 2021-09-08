import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


class LetterCounter:
    def __init__(self, values: List[int or float] = None):
        self.__letter_counter = None
        self.__is_letter_counter = False
        if values is not None:
            self.__fill_letter_counter(values=values)

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

        self.__is_letter_counter = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json object
        """
        if not self.__is_letter_counter:
            raise Exception("The values were not loaded!")
        return self.__letter_counter

    def set_values(self, values: List[int or float]):
        if not self.__is_letter_counter:
            if values is not None:
                self.__fill_letter_counter(values=values)

    def __fill_letter_counter(self, values: List[int or float]) -> None:
        """
        This method fill NormalDistribution class when we use values
        :param values: list of column values
        """
        self.__letter_counter = {}
        for string in values:
            for letter in string:
                if letter not in self.__letter_counter:
                    self.__letter_counter[letter] = 1
                else:
                    self.__letter_counter[letter] += 1
        self.__is_letter_counter = True


class StringIndicators:
    def __init__(self, values: List[int or float] = None, extended: bool = False):
        self.__min_len = None
        self.__mean_len = None
        self.__max_len = None
        self.letter_counter = None
        self.__use_letter_counter = extended
        self.__is_string_indicators = False  # Отвечает за наличие данных в классе NumericalIndicators
        self.__is_letter_counter = False  # Отвечает за наличие данных в классе NormalDistribution
        if values is not None:
            self.__fill_string_indicators(values=values,
                                          extended=extended)

    def get_min_len(self):
        """
        This method return minimal value of column
        :return Minimal value of column
        """
        return self.__min_len

    def get_max_len(self):
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        return self.__max_len

    def get_mean_len(self):
        """
        This method return mean value of column
        :return Mean value of column
        """
        return self.__mean_len

    def get_is_letter_counter(self) -> bool:
        return self.__is_letter_counter

    def get_letter_counter(self) -> LetterCounter:
        if self.__is_letter_counter:
            return self.letter_counter

    def get_is_string_indicators(self) -> bool:
        """
        This method return the current state of filling numerical indicators
        """
        return self.__is_string_indicators

    def set_values(self, values: List[int or float], extended: bool):
        """
        This method init the filling params of this class
        :param values: Values from DataSetColumn - values of column from the dataset
        :param extended: The switch responsible for calculating the indicators of the normal distribution
        :return: None
        """
        if not self.__is_string_indicators:
            if values is not None:
                self.__fill_string_indicators(values=values,
                                              extended=extended)

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
        self.__min_len = data["Minimal string length"]
        self.__max_len = data["Maximal string length"]
        self.__mean_len = data["Mean string length"]
        self.__is_string_indicators = True
        if "Normal distribution" in data:
            self.letter_counter = LetterCounter()
            self.__is_string_indicators = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json data
        """
        if not self.__is_string_indicators:
            raise Exception("The values were not loaded!")
        data = {"Minimal value": self.__min_len,
                "Maximal value": self.__max_len,
                "Mean value": self.__mean_len}
        if self.__use_letter_counter and self.__is_letter_counter:
            data['Letter counter'] = self.letter_counter.to_json()
        return data

    def __fill_string_indicators(self, values: List[int or float] or pd.DataFrame, extended: bool = False):
        """
        This method fill this class when we use values
        :param values: list of column values
        :param extended: The switch responsible for calculating the indicators of the normal distribution
        """
        values = values.tolist()
        values = [val for val in values if isinstance(val, str)]
        self.__min_len = min(values)
        self.__max_len = max(values)
        self.__mean_len = len("".join(values)) / len(values)
        self.__use_extended = extended
        self.__is_string_indicators = True
        if extended and not self.__is_letter_counter:
            self.letter_counter = LetterCounter()
            self.letter_counter.set_values(values=values)
            self.__is_letter_counter = True
