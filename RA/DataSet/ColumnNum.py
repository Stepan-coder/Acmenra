import math
import pandas as pd
from typing import List, Any
from RA.DataSet.ColumnType import *


class ColumnNum:
    def __init__(self, column: pd.Series, column_type: ColumnType):
        """
        This method init a class work
        :param column: The column of DataSet
        :param column_type: The type of DataSet column others
        """
        self.__column = column
        self.__column_type = column_type

    def __add__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method adds a value to a number (+)
        :param other: What needs to be added
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] += other
        return new_some.to_list()

    def __sub__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method subtracts the value from the number (-)
        :param other: What needs to be subtracted
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] -= other
        return new_some.to_list()

    def __mul__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method multiplies the value by a number (*)
        :param other: What should be multiplied
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] *= other
        return new_some.to_list()

    def __floordiv__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method divides the value by a number (//)
        :param other: What should be divided
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] //= other
        return new_some.to_list()

    def __div__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method divides the value by a number (/)
        :param other: What should be divided
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] /= other
        return new_some.to_list()

    def __mod__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method gets the remainder from dividing the value by a number (%)
        :param other: What should be divided
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] %= other
        return new_some.to_list()

    def __pow__(self, other: int or float) -> List[str]:
        """
        (For each cell in column)
        This method is to raise values to the power of a number (**)
        :param other: To what extent should it be raised
        :return: List[str]
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] **= other
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

    def __instancecheck__(self, instance: Any) -> bool:
        """
        This method checks is instance type is DataSet
        :param instance: Checked value
        :return: bool
        """
        return isinstance(instance, type(self))

    def __len__(self) -> int:
        """
        This method returns count of elements in column
        :return: int
        """
        return len(self.__column)

    @property
    def type(self) -> ColumnType:
        """
        This property returns a type of column
        :return: ColumnType
        """
        return self.__column_type

    def values(self) -> List[bool or int or float]:
        """
        This method returns column values as a list
        :return: list
        """
        return self.__column.to_list()

    def add(self, other: int or float):
        """
        (For each cell in column)
        This method adds a value to a number (+)
        :param other: What needs to be added
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] += other
        return self

    def sub(self, other: int or float):
        """
        (For each cell in column)
        This method subtracts the value from the number (-)
        :param other: What needs to be subtracted
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] -= other
        return self

    def mul(self, other: int or float):
        """
        (For each cell in column)
        This method multiplies the value by a number (*)
        :param other: What should be multiplied
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] *= int(other)
        return self

    def floordiv(self, other: int or float):
        """
        (For each cell in column)
        This method divides the value by a number (//)
        :param other: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        if other == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] //= other
        return self

    def div(self, other: int or float):
        """
        (For each cell in column)
        This method divides the value by a number (/)
        :param other: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        if other == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] /= other
        return self

    def mod(self, other: int or float):
        """
        (For each cell in column)
        This method gets the remainder from dividing the value by a number (%)
        :param other: What should be divided
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        if other == 0:
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] %= other
        return self

    def pow(self, other: int or float):
        """
        (For each cell in column)
        This method is to raise others to the value of a number (**)
        :param other: To what extent should it be raised
        :return: DataSetColumnNum
        """
        if not (isinstance(other, int) or isinstance(other, float)):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        for i in range(len(self.__column)):
            self.__column[i] **= other
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
