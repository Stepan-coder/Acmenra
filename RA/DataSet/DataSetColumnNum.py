import math
import pandas as pd

from typing import List


class DataSetColumnNum:
    def __init__(self, column: pd.Series, column_type: str):
        """
        This method init a class work
        :param column: The column of DataSet
        :param column_type: The type of DataSet column values
        """
        self.__column = column
        self.__column_type = column_type

    def __add__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method adds a value to a number (+)
        :param value: What needs to be added
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] += value
        return new_some.to_list()

    def __sub__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method subtracts the value from the number (-)
        :param value: What needs to be subtracted
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] -= value
        return new_some.to_list()

    def __mul__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method multiplies the value by a number (*)
        :param value: What should be multiplied
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] *= value
        return new_some.to_list()

    def __floordiv__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method divides the value by a number (//)
        :param value: What should be divided
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] //= value
        return new_some.to_list()

    def __div__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method divides the value by a number (/)
        :param value: What should be divided
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] /= value
        return new_some.to_list()

    def __mod__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method gets the remainder from dividing the value by a number (%)
        :param value: What should be divided
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] %= value
        return new_some.to_list()

    def __pow__(self, value: int or float) -> List[str]:
        """
        (For each cell in column)
        This method is to raise values to the power of a number (**)
        :param value: To what extent should it be raised
        :return: List[str]
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] **= value
        return new_some.to_list()

    def __round__(self, n: int = None):
        """
        (For each cell in column)
        This method rounds the value to the specified precision
        :param n: Count of characters
        :return: List[str]
        """
        if not isinstance(n, int):
            raise Exception("The type \'n\' must be \'int\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] = round(new_some[i], ndigits=n)
        return new_some.to_list()

    def __floor__(self):
        """
        (For each cell in column)
        This method rounds the value to the nearest smaller integer
        :return: List[str]
        """
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] = math.floor(new_some[i])
        return new_some.to_list()

    def __ceil__(self):
        """
        (For each cell in column)
        This method rounds the value to the nearest bigger integer
        :return: List[str]
        """
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] = math.ceil(new_some[i])
        return new_some.to_list()

    def __trunc__(self):
        """
        (For each cell in column)
        This method truncates the value to an integer
        :return: List[str]
        """
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] = math.trunc(new_some[i])
        return new_some.to_list()

    def add(self, value: int or float):
        """
        (For each cell in column)
        This method adds a value to a number (+)
        :param value: What needs to be added
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] += value
        return self

    def sub(self, value: int or float):
        """
        (For each cell in column)
        This method subtracts the value from the number (-)
        :param value: What needs to be subtracted
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] -= value
        return self

    def mul(self, value: int or float):
        """
        (For each cell in column)
        This method multiplies the value by a number (*)
        :param value: What should be multiplied
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] *= int(value)
        return self

    def floordiv(self, value: int or float):
        """
        (For each cell in column)
        This method divides the value by a number (//)
        :param value: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        if value == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] //= value
        return self

    def div(self, value: int or float):
        """
        (For each cell in column)
        This method divides the value by a number (/)
        :param value: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        if value == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] /= value
        return self

    def mod(self, value: int or float):
        """
        (For each cell in column)
        This method gets the remainder from dividing the value by a number (%)
        :param value: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        if value == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] %= value
        return self

    def pow(self, value: int or float):
        """
        (For each cell in column)
        This method is to raise values to the power of a number (**)
        :param value: To what extent should it be raised
        :return: DataSetColumnNum
        """
        if not (isinstance(value, int) or isinstance(value, float)):
            raise Exception("The type \'value\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] **= value
        return self

    def round(self, n: int = None):
        """
        (For each cell in column)
        This method rounds the value to the specified precision
        :param n: Count of characters
        :return: DataSetColumnNum
        """
        if not isinstance(n, int):
            raise Exception("The type \'n\' must be \'int\'")
        for i in range(len(self.__column)):
            self.__column[i] = round(self.__column[i], ndigits=int(n))
        return self

    def floor(self):
        """
        (For each cell in column)
        This method rounds the value to the nearest smaller integer
        :return: DataSetColumnNum
        """
        for i in range(len(self.__column)):
            self.__column[i] = math.floor(self.__column[i])
        return self

    def ceil(self):
        """
        (For each cell in column)
        This method rounds the value to the nearest bigger integer
        :return: DataSetColumnNum
        """
        for i in range(len(self.__column)):
            self.__column[i] = math.ceil(self.__column[i])
        return self

    def trunc(self):
        """
        (For each cell in column)
        This method truncates the value to an integer
        :return: DataSetColumnNum
        """
        for i in range(len(self.__column)):
            self.__column[i] = math.trunc(self.__column[i])
        return self

    def to_list(self) -> list:
        """
        This method returns column values as a list
        :return: list
        """
        return self.__column.to_list()
