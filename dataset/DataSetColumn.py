import math
import pandas as pd


class DataSetFieldAnalytics:
    def __init__(self,
                 column_name: str,
                 values: list = None,
                 normal_distribution: bool = False,
                 rounding_factor: int = 2):
        self.column_name = column_name
        self.values = values
        self.normal_distribution = normal_distribution
        self.rounding_factor = rounding_factor
        if self.values is not None:  # If we have received the values, then we count the statistics
            self.get_column_info()

    def __str__(self):
        """
        Displaying information about the column
        """
        text = f"DataSetField: \'{self.column_name}\'\n"
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

    def get_column_info(self):
        if self.values is not None:  # If we have received the values, then we count the statistics
            self.field_type = type(self.values[0]).__name__
            self.count = len(self.values)
            self.count_unique = len(list(set(self.values)))
            self.nan_count = self._get_nan_count()
            if self.field_type.startswith("int") or self.field_type.startswith("float"):  # If it is a numeric format
                self.min = min(self.values)
                self.max = max(self.values)
                self.average = sum(self.values) / self.count
                if self.normal_distribution:  # Exporting additional statistics
                    math_rasp = self._get_math_rasp()
                    self.math_moda = self._get_math_moda(math_rasp)
                    self.math_wait = self._get_math_wait(math_rasp)
                    self.math_dispersion = self._get_math_dispersion(math_rasp)
                    if self.math_dispersion >= 0:
                        self.math_sigma = math.sqrt(self.math_dispersion)
                    else:
                        self.math_sigma = -1 * math.sqrt(abs(self.math_dispersion))

    def get_from_json(self, data):
        self.column_name = data["column_name"]
        self.field_type = data["type"]
        self.count = data["count"]
        self.count_unique = data["count_unique"]
        self.nan_count = data["count_NaN"]
        if "numerical_indicators" in data:
            self.min = data["numerical_indicators"]["min"]
            self.max = data["numerical_indicators"]["max"]
            self.average = data["numerical_indicators"]["average"]
        if "normal_distribution" in data:
            self.math_moda = data["normal_distribution"]["moda"]
            self.math_wait = data["normal_distribution"]["wait"]
            self.math_dispersion = data["normal_distribution"]["dispersion"]
            self.math_sigma = data["normal_distribution"]["sigma"]

    def to_json(self) -> dict:
        """
        This method export DataSet statistics as json
        :return: dict[column name, json statistic]
        """
        data = {"column_name": self.column_name,
                "type": self.field_type,
                "count": self.count,
                "count_unique": self.count_unique,
                "count_NaN": self.nan_count
                }
        if self.field_type.startswith("int") or self.field_type.startswith("float"):
            data["numerical_indicators"] = {"min": self.min,
                                            "max": self.max,
                                            "average": self.average
                                            }
            if self.normal_distribution:
                data["normal_distribution"] = {
                    "moda": self.math_moda,
                    "wait": round(self.math_wait, self.rounding_factor),
                    "dispersion": round(self.math_dispersion, self.rounding_factor),
                    "sigma": round(self.math_sigma, self.rounding_factor),
                    "moda-3Xsigma": round(self.math_wait - 3 * self.math_sigma, self.rounding_factor),
                    "moda-2Xsigma": round(self.math_wait - 2 * self.math_sigma, self.rounding_factor),
                    "moda-1Xsigma": round(self.math_wait - 1 * self.math_sigma, self.rounding_factor),
                    "moda+1Xsigma": round(self.math_wait + 1 * self.math_sigma, self.rounding_factor),
                    "moda+2Xsigma": round(self.math_wait + 2 * self.math_sigma, self.rounding_factor),
                    "moda+3Xsigma": round(self.math_wait + 3 * self.math_sigma, self.rounding_factor)

                }
        return {self.column_name: data}

    def _get_nan_count(self) -> int:
        """
        This method calculate count of NaN values
        :return: int
        """
        nan_cnt = 0
        for value in self.values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt

    def _get_math_rasp(self):
        """
        This method calculates the frequency of list values as a percentage
        :return: dict
        """
        math_rasp = {}
        for value in self.values:
            if value in list(math_rasp):
                math_rasp[value] += (1 / len(self.values))
            else:
                math_rasp[value] = (1 / len(self.values))
        return math_rasp

    def _get_math_dispersion(self, math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical variance
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait = self._get_math_wait(math_rasp_dict)
        math_wait_x2 = math_wait * math_wait
        math_wait_2 = self._get_math_wait_2(math_rasp_dict)
        return math_wait_2 - math_wait_x2

    @staticmethod
    def _get_math_moda(math_rasp_dict: dict):
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
    def _get_math_wait(math_rasp_dict: dict) -> float:
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
    def _get_math_wait_2(math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical expectation squared
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait_2 = 0
        for md in math_rasp_dict:
            math_wait_2 += float(md) * float(md) * math_rasp_dict[md]
        return math_wait_2
