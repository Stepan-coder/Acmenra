import pandas as pd
from typing import List, Any
from RA.DataSet.ColumnType import *


class ColumnStr:
    def __init__(self, column: pd.Series, column_type: ColumnType):
        self.__column = column
        self.__column_type = column_type

    def __add__(self, other: str) -> List[str]:
        """
        This method adds a other value to each column cell
        :param other: What needs to be added
        """
        if not isinstance(other, str):
            raise Exception("The type of \'other\' must be \'str\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] = str(new_some[i]) + other
        return new_some.to_list()

    def __mul__(self, other: int or float) -> List[str]:
        """
        This method repeats the value of each cell other count
        :param other: What needs to be added
        """
        if not isinstance(other, int or float):
            raise Exception("The type of \'other\' must be \'int\' or \'float\'")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] *= other
        return new_some.to_list()

    def __instancecheck__(self, instance: Any) -> bool:
        """
        This method checks is instance type is DataSet
        :param instance: Checked value
        """
        return isinstance(instance, type(self))

    def __len__(self) -> int:
        """
        This method returns count of elements in column
        """
        return len(self.__column)

    @property
    def type(self) -> ColumnType:
        """
        This property returns a type of column
        """
        return self.__column_type

    def values(self) -> List[bool or int or float]:
        """
        This method returns column values as a list
        """
        return self.__column.to_list()

    def add(self, other: str) -> None:
        """
        This method adds a other value to each column cell
        :param other: What needs to be added
        """
        if not isinstance(other, str):
            raise Exception("The type of \'other\' must be \'str\'")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]) + other

    def mul(self, other: int or float) -> None:
        """
        This method repeats the value of each cell other count
        :param other: What needs to be added
        """
        if not isinstance(other, int or float):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] *= int(other)

    def capitalize(self) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a capitalized version of String, i.e. make the first character
        have upper case and the rest lower case.
        """
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).capitalize()
        return self

    def title(self) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return True if S is a titlecased string and there is at least one
        character in S, i.e. upper- and titlecase characters may only
        follow uncased characters and lowercase characters only cased ones.
        Return False otherwise.
        """
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).title()
        return self

    def upper(self) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a copy of S converted to uppercase.
        """
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).upper()
        return self

    def lower(self) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a copy of B with all ASCII characters converted to lowercase.
        """
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).lower()
        return self

    def replace(self, old_string: str, new_string: str) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a copy with all occurrences of substring old replaced by new.
          count
            Maximum number of occurrences to replace.
            -1 (the default value) means replace all occurrences.
        If the optional argument count is given, only the first count occurrences are
        replaced.
        :param old_string: Substring in the string to be replaced
        :param new_string: The substring to be replaced by
        """
        if old_string is not None and not isinstance(old_string, str):
            raise Exception("The type of \'old_string\' must be \'str\'")
        if new_string is not None and not isinstance(new_string, str):
            raise Exception("The type of \'new_string\' must be \'str\'")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).replace(old_string, new_string)
        return self

    def strip(self, chars: str = None) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Strip leading and trailing bytes contained in the argument.
        If the argument is omitted or None, strip leading and trailing ASCII whitespace.
        :param chars: The string to be removed from the beginning and end of the string
        """
        if chars is not None and not isinstance(chars, str):
            raise Exception("The type of \'chars\' must be \'str\'")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).strip(chars)
        return self

    def lstrip(self, chars: str = None) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a copy of the string S with trailing whitespace removed.
        If chars is given and not None, remove characters in chars instead.
        :param chars: The string to be removed from the beginning of the string
        """
        if chars is not None and not isinstance(chars, str):
            raise Exception("The type of \'chars\' must be \'str\'")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).lstrip(chars)
        return self

    def rstrip(self, chars: str = None) -> 'DataSetColumnStr':
        """
        (For each cell in column)
        Return a copy of the string S with trailing whitespace removed.
        If chars is given and not None, remove characters in chars instead.
        :param chars: The string to be removed from the end of the string
        """
        if chars is not None and not isinstance(chars, str):
            raise Exception("The type of \'chars\' must be \'str\'")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).rstrip(chars)
        return self
