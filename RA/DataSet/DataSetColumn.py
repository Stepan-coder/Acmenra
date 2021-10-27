import pandas as pd


class DataSetColumnStr:
    def __init__(self, column: pd.Series, type: str):
        self.__column = column
        self.__type = type

    # def __add__(self, other):
    #     if isinstance(other, str) and self.__type.startswith("str")
    #     new_some = self.some.copy()
    #     for i in range(len(self.some)):
    #         new_some[i] += other
    #     return new_some.to_list()

    # def __mul__(self, other):

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

    def upper(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).upper()
        return self

    def lower(self):
        for i in range(len(self.__column)):
            self.__column[i] = str(self.__column[i]).lower()
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