import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from typing import Dict, List
from prettytable import PrettyTable


class LetterDistribution:
    def __init__(self, values: List[int or float] = None):
        self.__letter_distribution = None
        self.__is_letter_distribution = False
        if values is not None:
            self.__fill_letter_distribution(values=values)

    def __str__(self):
        table = PrettyTable()
        table.title = f"\"LetterDistribution\""
        table.field_names = ["Indicator", "Value"]
        if self.__is_letter_distribution:
            letter_distr = self.get_letter_distribution()
            for ld in letter_distr:
                table.add_row([ld, letter_distr[ld]])
        return str(table)

    def get_letter_distribution(self) -> Dict[int or float, float]:
        """
        This method return mathematical distribution dict
        """
        if self.__is_letter_distribution is not None:
            return self.__letter_distribution
        else:
            raise Exception("The data has not been loaded yet!")

    def from_json(self, data) -> None:
        """
        This method load class fields from json
        :param data: Input json object
        :return: None
        """
        self.__letter_distribution = data
        self.__is_letter_distribution = True

    def to_json(self):
        """
        This method export class NormalDistribution to json object
        :return: json object
        """
        if not self.__is_letter_distribution:
            raise Exception("The values were not loaded!")
        return self.__letter_distribution

    def set_values(self, values: List[int or float]) -> None:
        """
        This method set values to the LetterDistribution class
        :param values:
        :return: None
        """
        if not self.__is_letter_distribution:
            if values is not None:
                self.__fill_letter_distribution(values=values)

    def __fill_letter_distribution(self, values: List[int or float]) -> None:
        """
        This method fill LetterCounter class when we use values
        :param values: list of column values
        :return None
        """
        self.__letter_distribution = {}
        for string in tqdm(values):
            for letter in string:
                if letter not in self.__letter_distribution:
                    self.__letter_distribution[letter] = 1
                else:
                    self.__letter_distribution[letter] += 1
        self.__letter_distribution = dict(sorted(self.__letter_distribution.items(), key=lambda x: x[1], reverse=True))
        self.__is_letter_distribution = True


class StringIndicators:
    def __init__(self, extended: bool, values: List[int or float] = None):
        self.__min_len = None
        self.__mean_len = None
        self.__max_len = None
        self.__letter_counter = None
        self.__use_letter_counter = extended
        self.__is_string_indicators = False  # Отвечает за наличие данных в классе NumericalIndicators
        self.__is_letter_counter = False  # Отвечает за наличие данных в классе NormalDistribution
        if values is not None:
            self.__fill_string_indicators(values=values,
                                          extended=extended)

    def __str__(self):
        table = PrettyTable()
        table.title = f"\"StringIndicators\""
        table.field_names = ["Indicator", "Value"]
        if self.__is_string_indicators:
            table.add_row(["Minimal string length", self.get_min()])
            table.add_row(["Mean string length", self.get_mean()])
            table.add_row(["Maximal string length", self.get_max()])
        return str(table)

    def set_values(self, values: List[int or float], extended: bool) -> None:
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

    def get_min(self):
        """
        This method return minimal value of column
        :return Minimal value of column
        """
        return self.__min_len

    def get_max(self):
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        return self.__max_len

    def get_mean(self):
        """
        This method return mean value of column
        :return Mean value of column
        """
        return self.__mean_len

    def get_is_letter_counter(self) -> bool:
        return self.__is_letter_counter

    def get_letter_counter(self) -> LetterDistribution:
        if self.__is_letter_counter:
            return self.__letter_counter

    def get_is_string_indicators(self) -> bool:
        """
        This method return the current state of filling numerical indicators
        """
        return self.__is_string_indicators

    def get_from_json(self, data: dict) -> None:
        """
        This method load NumericalIndicators indicators from json
        :param data: Incoming data in json format
        :return: None
        """
        required_fields = ["Minimal string length", "Maximal string length", "Mean string length"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__min_len = data["Minimal string length"]
        self.__max_len = data["Maximal string length"]
        self.__mean_len = data["Mean string length"]
        self.__is_string_indicators = True
        if "Letter counter" in data:
            self.__letter_counter = LetterDistribution()
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

    def __fill_string_indicators(self, values: List[int or float] or pd.DataFrame, extended: bool) -> None:
        """
        This method fill this class when we use values
        :param values: list of column values
        :param extended: The switch responsible for calculating the indicators of the normal distribution
        :return None
        """
        try:
            values = values.tolist()
        except:
            pass
        values = [val for val in values if isinstance(val, str)]
        self.__min_len = min([len(v) for v in values])
        self.__max_len = max([len(v) for v in values])
        self.__mean_len = len("".join(values)) / len(values)
        self.__use_extended = extended
        self.__is_string_indicators = True
        if extended and not self.__is_letter_counter:
            self.letter_counter = LetterDistribution()
            self.letter_counter.set_values(values=values)
            self.__is_letter_counter = True
