import pandas as pd


class DataSetColumnStr:
    def __init__(self, column: pd.Series, type: str):
        self.__column = column
        self.__type = type

    def __add__(self, other):
        if isinstance(other, str):
            raise Exception("Error")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] += other
        return new_some.to_list()

    def __iadd__(self, other):
        self.add(other=other)

    def __mul__(self, other):
        if isinstance(other, str):
            raise Exception("Error")
        new_some = self.__column.copy()
        for i in range(len(self.__column)):
            new_some[i] += other
        return new_some.to_list()

    def 
    def add(self, other) -> None:
        if not isinstance(other, str):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] += other

    def mul(self, other) -> None:
        if not isinstance(other, int or float):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] *= int(other)

    def capitalize(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).capitalize()
        return self

    def title(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).title()
        return self

    def upper(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).upper()
        return self

    def lower(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).lower()
        return self

    def replace(self, old_string: str, new_string: str):
        if old_string is not None and not isinstance(old_string, str):
            raise Exception("Error")
        if new_string is not None and not isinstance(new_string, str):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).replace(old_string, new_string)
        return self

    def strip(self, chars: str = None):
        if chars is not None and not isinstance(chars, str):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).strip(chars)
        return self

    def lstrip(self, chars: str = None):
        if chars is not None and not isinstance(chars, str):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).lstrip(chars)
        return self

    def rstrip(self, chars: str = None):
        if chars is not None and not isinstance(chars, str):
            raise Exception("Error")
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).rstrip(chars)
        return self