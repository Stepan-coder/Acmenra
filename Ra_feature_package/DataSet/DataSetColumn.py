import math
import pandas as pd

from Ra_feature_package.DataSet.DataSetColumnNumStat import *


class DataSetColumn:
    def __init__(self,
                 column_name: str,
                 values: list = None,
                 normal_distribution: bool = False,
                 rounding_factor: int = 2):
        self.__column_name = column_name
        self.__values = values
        self.__use_normal_distribution = normal_distribution
        self.__rounding_factor = rounding_factor

        self.__is_num_stat = False  # Отвечает за наличие числовых расчётов
        self.__field_type = None
        self.__count = None
        self.__count_unique = None
        self.__nan_count = None
        self.num_stat = None
        if values is not None:
            self.__fill_dataset_column(values=values,
                                       normal_distribution=normal_distribution,
                                       rounding_factor=rounding_factor)

    def __str__(self):
        """
        Displaying information about the column
        """
        text = f"DataSetColumn: \'{self.column_name}\'\n"
        text += f"  -Name: \'{self.column_name}\'\n"
        text += f"  -Type: \'{self.field_type}\'\n"
        text += f"  -Count: {self.count}\n"
        text += f"  -Count unique: {self.count_unique}\n"
        text += f"  -Count NaN: {self.nan_count}\n"
        if self.field_type.startswith("int") or self.field_type.startswith("float"):
            text += f"  Numerical indicators:\n"
            text += f"      -Min: {self.min}\n"
            text += f"      -Max: {self.max}\n"
            text += f"      -Average: {round(self.average, self.rounding_factor)}\n"
            if self.normal_distribution:
                text += f"  Normal distribution:\n"
                text += f"      -Moda: {self.math_moda}\n"
                text += f"      -Wait: {round(self.math_wait, self.rounding_factor)}\n"
                text += f"      -Dispersion: {round(self.math_dispersion, self.rounding_factor)}\n"
                text += f"      -Sigma: {round(self.math_sigma, self.rounding_factor)}\n"
                text += "      -Moda - 3 * Sigma: {0}\n".format(round(self.math_wait - 3 * self.math_sigma,
                                                                      self.rounding_factor))
                text += "      -Moda - 2 * Sigma: {0}\n".format(round(self.math_wait - 2 * self.math_sigma,
                                                                      self.rounding_factor))
                text += "      -Moda - 1 * Sigma: {0}\n".format(round(self.math_wait - 1 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 1 * Sigma: {0}\n".format(round(self.math_wait + 1 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 2 * Sigma: {0}\n".format(round(self.math_wait + 2 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 3 * Sigma: {0}\n".format(round(self.math_wait + 3 * self.math_sigma),
                                                                self.rounding_factor)
        return text

    def get_column_name(self) -> str:
        return self.__column_name

    def get_is_num_stat(self) -> bool:
        return self.__is_num_stat

    def get_values(self):
        return self.__values

    def get_num_stat(self) -> NumericalIndicators:
        return self.num_stat

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
        self.__field_type = data["type"]
        self.__count = data["count"]
        self.__count_unique = data["count_unique"]
        self.__nan_count = data["count_NaN"]
        self.__values = values
        if "Numerical indicators" in data:
            self.num_stat: NumericalIndicators = NumericalIndicators()
            self.num_stat.get_from_json(data=data["Numerical indicators"])
            self.__is_num_stat = True

    def to_json(self) -> dict:
        """
        This method export DataSet statistics as json
        :return: dict[column name, json statistic]
        """
        if self.__values is not None:
            data = {"column_name": self.__column_name,
                    "type": self.__field_type,
                    "count": self.__count,
                    "count_unique": self.__count_unique,
                    "count_NaN": self.__nan_count}
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

    def __fill_dataset_column(self,
                              values: list,
                              normal_distribution: bool = False,
                              rounding_factor: int = 2):
        """
        This method fill this class when we use values
        :param values: list of column values
        :param normal_distribution: The switch responsible for calculating the indicators of the normal distribution
        """
        self.__values = values
        self.__field_type = type(self.__values[0]).__name__
        self.__count = len(self.__values)
        self.__count_unique = len(list(set(self.__values)))
        self.__nan_count = self.__get_nan_count()
        if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
            self.num_stat = NumericalIndicators()
            self.num_stat.set_values(values=self.__values,
                                     normal_distribution=self.__use_normal_distribution)
            self.__is_num_stat = True

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


