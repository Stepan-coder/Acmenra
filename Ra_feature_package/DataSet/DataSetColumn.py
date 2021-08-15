import math
import pandas as pd

from Ra_feature_package.DataSet.DataSetColumnNumStat import *

"""
TODO
str - сделать красивой табличкой
"""

class DataSetColumn:
    def __init__(self,
                 column_name: str,
                 values: list = None,
                 categorical: int = 25,
                 normal_distribution: bool = False,
                 rounding_factor: int = 2):
        self.__column_name = column_name
        self.__values = None
        self.__count = None
        self.__count_unique = None
        self.__nan_count = None
        self.__field_type = None
        self.__field_dtype = None

        self.__use_normal_distribution = normal_distribution
        self.__rounding_factor = rounding_factor

        self.__is_num_stat = False  # Отвечает за наличие числовых расчётов
        self.__num_stat = None

        if values is not None:
            self.__fill_dataset_column(values=values,
                                       categorical=categorical,
                                       normal_distribution=normal_distribution,
                                       rounding_factor=rounding_factor)

    def __fill_dataset_column(self,
                              values: list,
                              categorical: int = 25,
                              normal_distribution: bool = False,
                              rounding_factor: int = 2):
        """
        This method fill this class when we use values
        :param values: list of column values
        :param normal_distribution: The switch responsible for calculating the indicators of the normal distribution
        """
        self.__values = values
        self.__count = len(self.__values)
        self.__count_unique = len(list(set(self.__values)))
        self.__field_type = type(self.__values[0]).__name__
        self.__field_dtype = "variable" if self.__count_unique >= categorical else "categorical"
        self.__nan_count = self.__get_nan_count()
        if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
            self.__num_stat = NumericalIndicators()
            self.__num_stat.set_values(values=self.__values,
                                       normal_distribution=self.__use_normal_distribution)
            self.__is_num_stat = True

    def __str__(self):
        """
        Displaying information about the column
        """
        text = f"DataSetColumn: \'{self.__column_name}\'\n"
        text += f"  -Name: \'{self.__column_name}\'\n"
        text += f"  -Type: \'{self.__field_type}\'\n"
        text += f"  -Count: {self.__count}\n"
        text += f"  -Count unique: {self.__count_unique}\n"
        text += f"  -Count NaN: {self.__nan_count}\n"
        return text

    def get_column_name(self) -> str:
        return self.__column_name

    def get_count(self) -> int:
        if self.__count is None:
            raise Exception("The values were not loaded!")
        return self.__count

    def get_count_unique(self) -> int:
        if self.__count_unique is None:
            raise Exception("The values were not loaded!")
        return self.__count_unique

    def get_nan_count(self) -> int:
        if self.__nan_count is None:
            raise Exception("The values were not loaded!")
        return self.__nan_count

    def get_type(self):
        if self.__field_type is None:
            raise Exception("The values were not loaded!")
        return self.__field_type

    def get_dtype(self):
        if self.__field_dtype is None:
            raise Exception("The values were not loaded!")
        return self.__field_dtype

    def get_min(self) -> int or float or bool:
        """
        This method return minimal value of column
        :return Minimal value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_min()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_max(self) -> int or float or bool:
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_max()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_average(self) -> int or float or bool:
        """
        This method return maximal value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_average()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_is_num_stat(self) -> bool:
        return self.__is_num_stat

    def get_values(self):
        return self.__values

    def get_num_stat(self) -> NumericalIndicators:
        return self.__num_stat

    def get_from_json(self, data: dict, values: dict) -> None:
        """
        This method load DataSet indicators from json
        :param data:
        :param values:
        :return: None
        """
        required_fields = ["column_name", "type", "count", "count_unique", "count_NaN"]
        for rf in required_fields:
            if rf not in data:
                raise Exception("The resulting json file does not contain required arguments! Try another file.")
        self.__column_name = data["column_name"]
        self.__values = values
        self.__count = data["count"]
        self.__count_unique = data["count_unique"]
        self.__nan_count = data["count_NaN"]
        self.__field_type = data["type"]
        self.__field_dtype = data["dtype"]

        if "Numerical indicators" in data:
            self.__num_stat: NumericalIndicators = NumericalIndicators()
            self.__num_stat.get_from_json(data=data["Numerical indicators"])
            self.__is_num_stat = True

    def to_json(self) -> dict:
        """
        This method export DataSet statistics as json
        :return: dict[column name, json statistic]
        """
        if self.__values is not None:
            data = {"column_name": self.__column_name,
                    "count": self.__count,
                    "count_unique": self.__count_unique,
                    "count_NaN": self.__nan_count,
                    "type": self.__field_type,
                    "dtype": self.__field_dtype}
            if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
                if not self.num_stat.get_is_numerical_indicators():
                    self.num_stat = NumericalIndicators()
                    self.num_stat.set_values(values=self.__values,
                                             normal_distribution=self.__use_normal_distribution)
                    self.__is_num_stat = True
                data["Numerical indicators"] = self.num_stat.to_json()
            return {self.__column_name: data}
        else:
            raise Exception("The values were not loaded!")

    def __get_nan_count(self) -> int:
        """
        This method calculate count of NaN values
        :return: Count of NaN values in this column
        """
        nan_cnt = 0
        for value in self.__values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt


