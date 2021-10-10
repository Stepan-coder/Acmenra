import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from typing import Dict, List
from prettytable import PrettyTable


class StringStatistics:
    def __init__(self, values: List[int or float] = None):
        self.__strings_distribution = None
        self.__letters_distribution = None
        self.__is_string_statistics = False
        if values is not None:
            self.__fill_letter_distribution(values=values)

    def __str__(self):
        table = PrettyTable()
        table.title = f"\"LetterDistribution\""
        table.field_names = ["Indicator", "Value"]
        if self.__is_string_statistics:
            letter_distr = self.get_distribution()
            for ld in letter_distr:
                table.add_row([ld, letter_distr[ld]])
        return str(table)

    def get_distribution(self) -> Dict[int or float, float]:
        """
        This method return mathematical distribution dict
        """
        if not self.__is_string_statistics:
            raise Exception("The data has not been loaded yet!")
        return self.__strings_distribution

    def get_letters_distribution(self) -> Dict[int or float, float]:
        """
        This method return mathematical distribution dict
        """
        if not self.__is_string_statistics:
            raise Exception("The data has not been loaded yet!")
        return self.__letters_distribution

    def from_json(self, data) -> None:
        """
        This method load class fields from json
        :param data: Input json object
        :return: None
        """
        self.__letters_distribution = data
        self.__is_string_statistics = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json object
        """
        if not self.__is_string_statistics:
            raise Exception("The values were not loaded!")
        return self.__is_string_statistics

    def set_values(self, values: List[int or float]) -> None:
        """
        This method set values to the LetterDistribution class
        :param values:
        :return: None
        """
        if not self.__is_string_statistics:
            if values is not None:
                self.__fill_letter_distribution(values=values)

    def __fill_letter_distribution(self, values: List[int or float]) -> None:
        """
        This method fill LetterCounter class when we use values
        :param values: list of column values
        :return None
        """
        self.__letters_distribution = {}
        self.__strings_distribution = {}
        for string in tqdm(values):
            if string not in self.__strings_distribution:
                self.__strings_distribution[string] = 1
            else:
                self.__strings_distribution[string] += 1
            for letter in string:
                if letter not in self.__letters_distribution:
                    self.__letters_distribution[letter] = 1
                else:
                    self.__letters_distribution[letter] += 1
        self.__letters_distribution = dict(sorted(self.__letters_distribution.items(),
                                                  key=lambda x: x[1],
                                                  reverse=True))
        self.__strings_distribution = dict(sorted(self.__strings_distribution.items(),
                                                  key=lambda x: x[1],
                                                  reverse=True))
        self.__is_string_statistics = True


class StringIndicators:
    def __init__(self, values: List[int or float], extended: bool):
        values = [val for val in values if isinstance(val, str)]
        self.__min_len = min([len(v) for v in values])
        self.__min_val = min(values, key=len)
        self.__max_len = max([len(v) for v in values])
        self.__max_val = max(values, key=len)
        self.__mean_len = len("".join(values)) / len(values)
        self.__letter_counter = None
        self.__is_extended = extended
        self.__is_letter_counter = False
        if extended:
            self.__letter_counter = StringStatistics()
            self.__letter_counter.set_values(values=values)
            self.__is_letter_counter = True

    def __str__(self):
        table = PrettyTable()
        table.title = f"\"StringIndicators\""
        table.field_names = ["Indicator", "Value"]
        if self.__is_string_indicators:
            table.add_row(["Minimal string", self.get_min_value()])
            table.add_row(["Minimal string length", self.get_min()])
            table.add_row(["Mean string length", self.get_mean()])
            table.add_row(["Maximal string", self.get_max_value()])
            table.add_row(["Maximal string length", self.get_max()])
        return str(table)

    def get_min(self) -> int:
        """
        This method return minimal len of string in column
        :return int
        """
        return self.__min_len

    def get_min_value(self):
        """
        This method return minimal string in column
        :return Minimal value of column
        """
        return self.__min_val

    def get_max(self):
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        return self.__max_len

    def get_max_value(self):
        """
        This method return maximal string in column
        :return Maximal value of column
        """
        return self.__max_val

    def get_mean(self):
        """
        This method return mean value of column
        :return Mean value of column
        """
        return self.__mean_len

    def get_is_extended(self) -> bool:
        return self.__is_letter_counter

    def get_letter_counter(self) -> StringStatistics:
        if self.__is_letter_counter:
            return self.__letter_counter

    def get_from_json(self, data: dict) -> None:
        """
        This method load NumericalIndicators indicators from json
        :param data: Incoming data in json format
        :return: None
        """
        required_fields = ["Minimal string",
                           "Minimal string length",
                           "Mean string length",
                           "Maximal string",
                           "Maximal string length"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__min_val = data["Minimal string"]
        self.__max_val = data["Maximal string"]
        self.__min_len = data["Minimal string length"]
        self.__max_len = data["Maximal string length"]
        self.__mean_len = data["Mean string length"]
        self.__is_string_indicators = True
        if "Letter counter" in data:
            self.__letter_counter = StringStatistics()
            self.__letter_counter.from_json(data["Letter counter"])
            self.__use_letter_counter = True
            self.__is_letter_counter = True

    def to_json(self) -> dict:
        """
        This method export class NormalDistribution to json object
        :return: json data
        """
        if not self.__is_string_indicators:
            raise Exception("The values were not loaded!")
        data = {"Minimal string length": self.__min_len,
                "Maximal string length": self.__max_len,
                "Mean string length": self.__mean_len}
        if self.__use_letter_counter and self.__is_letter_counter:
            data['Letter counter'] = self.__letter_counter.to_json()
        return data

