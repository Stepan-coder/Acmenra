from prettytable import PrettyTable
from RA.DataSet.DataSetColumnStrStat import *


class DataSetColumnStr:
    def __init__(self,
                 column_name: str,
                 values: list,
                 extended: bool,
                 categorical: int = 25) -> None:
        self.__column_name = column_name
        self.__is_extended = extended
        self.__count = len(values)  # Указываем явно, потому что этот класс не должен хранить все значения с колонки
        self.__count_unique = len(list(set(values)))
        self.__field_type = self.get_column_type(values=values)
        self.__field_dtype = "variable" if self.__count_unique >= categorical else "categorical"
        self.__nan_count = self.__get_nan_count(values=values)
        self.__str_stat = StringIndicators(values=values, extended=extended)

    def __str__(self):
        table = PrettyTable()
        table.title = f"{'Simple' if not self.__is_extended else 'Extended'} Column \"{self.__column_name}\""
        table.field_names = ["Indicator", "Value"]
        table.add_row(["Сolumn name", self.get_column_name()])
        table.add_row(["Type", self.get_type()])
        table.add_row(["DType", self.get_dtype()])
        table.add_row(["Count", self.__count])
        table.add_row(["Count unique", self.get_unique_count()])
        table.add_row(["NaN count", self.get_nan_count()])
        table.add_row(["String indicators", "".join(len("String indicators") * [" "])])
        table.add_row(["Min val", self.get_min()])
        table.add_row(["Max val", self.get_max()])

        if self.get_str_stat().get_letter_counter().get_distribution():
            table.add_row(["Normal Distribution", "".join(len("Normal Distribution") * [" "])])
        return str(table)

    def __len__(self) -> int:
        """
        This method returns the len[count] of values in this column
        :return: int
        """
        return self.__count

    def get_str_stat(self) -> StringIndicators:
        return self.__str_stat

    def get_is_extended(self):
        return self.__is_extended

    def get_column_name(self) -> str:
        """
        This method return the name of current column
        :return: Name of column
        """
        return self.__column_name

    def get_count(self) -> int:
        """"
        This method returns count of values in this column
        :return: Count of values
        """
        return self.__count

    def get_unique_count(self) -> int:
        """
        This method returns count of unique values in this column
        :return: Count of unique values
        """
        return self.__count_unique

    def get_nan_count(self) -> int:
        """
        This method returns count of NaN values in this column
        :return: Count of NaN values
        """
        return self.__nan_count

    def get_type(self) -> str:
        """
        This method returns type of column
        :return: Type of column
        """
        if self.__field_type is None:
            raise Exception("The values were not loaded!")
        return self.__field_type

    def get_dtype(self):
        """
        This method returns the real type of column
        :return: Real type of column
        """
        return self.__field_dtype

    def get_min(self) -> int or float or bool:
        """
        This method return minimal str len in column
        :return Minimal value of column
        """
        return self.__str_stat.get_min()

    def get_min_value(self) -> int or float or bool:
        """
        This method return minimal value of column
        :return Minimal value of column
        """
        return self.__str_stat.get_min_value()

    def get_max(self) -> int or float or bool:
        """
        This method return maximal str len of column
        :return Maximal value of column
        """
        return self.__str_stat.get_max()

    def get_max_value(self) -> int or float or bool:
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        return self.__str_stat.get_max_value()

    def get_mean(self) -> int or float:
        """
        This method return maximal value of column
        :return Mean value of column
        """
        return self.__str_stat.get_mean()

    def get_values_distribution(self) -> Dict[str, float]:
        """
        This method returns the percentage of values in the column
        :return Dict[bool or float or int or str, float]
        """
        if not self.get_str_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__str_stat.get_letter_counter().get_distribution()

    def get_letters_distribution(self) -> Dict[str, float]:
        """
        This method returns the percentage of values in the column
        :return Dict[bool or float or int or str, float]
        """
        if not self.get_str_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__str_stat.get_letter_counter().get_letters_distribution()


    def get_column_type(self, values: list) -> str:
        types = []
        for i in range(len(values)):
            types.append(type(values[i]).__name__)
        types = list(set(types))
        if len(types) == 1:
            return types[0]
        else:
            if len(types) == 2 and 'str' in types:
                return 'str'
            return "object"

    # def get_from_json(self, data: dict, values: dict) -> None:
    #     """
    #     This method load DataSet indicators from json
    #     :param data:
    #     :param values:
    #     :return: None
    #     """
    #     required_fields = ["column_name", "type", "count", "count_unique", "count_NaN"]
    #     for rf in required_fields:
    #         if rf not in data:
    #             raise Exception("The resulting json file does not contain required arguments! Try another file.")
    #     self.__column_name = data["column_name"]
    #     self.__values = values
    #     self.__count = data["count"]
    #     self.__count_unique = data["count_unique"]
    #     self.__nan_count = data["count_NaN"]
    #     self.__field_type = data["type"]
    #     self.__field_dtype = data["dtype"]
    #     if "Numerical indicators" in data:
    #         self.__num_stat: NumericalIndicators = NumericalIndicators(extended=False)
    #         self.__num_stat.get_from_json(data=data["Numerical indicators"])
    #         self.__is_num_stat = True
    #     if "String Indicators" in data:
    #         self.__str_stat: StringIndicators = StringIndicators(extended=False)
    #         self.__str_stat.get_from_json(data=data["String Indicators"])
    #         self.__is_str_stat = True
    #
    # def to_json(self) -> Dict[str, dict]:
    #     """
    #     This method export DataSet statistics as json
    #     :return: Dict[str, json]
    #     """
    #     if self.__values is not None:
    #         data = {"column_name": self.__column_name,
    #                 "count": self.__count,
    #                 "count_unique": self.__count_unique,
    #                 "count_NaN": self.__nan_count,
    #                 "type": self.__field_type,
    #                 "dtype": self.__field_dtype}
    #         if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
    #             if not self.__num_stat.get_is_numerical_indicators():
    #                 self.__num_stat = NumericalIndicators(extended=self.__use_extended)
    #                 self.__num_stat.set_values(values=self.__values,
    #                                            extended=self.__use_extended)
    #                 self.__is_num_stat = True
    #             data["Numerical indicators"] = self.__num_stat.to_json()
    #         elif self.__field_type.startswith("str"):
    #             if not self.__str_stat.get_is_string_indicators():
    #                 self.__str_stat = StringIndicators(extended=self.__use_extended)
    #                 self.__str_stat.set_values(values=self.__values,
    #                                            extended=self.__use_extended)
    #                 self.__is_str_stat = True
    #             data["String Indicators"] = self.__str_stat.to_json()
    #         return {self.__column_name: data}
    #     else:
    #         raise Exception("The values were not loaded!")

    def __get_nan_count(self, values: list) -> int:
        """
        This method calculate count of NaN values
        :return: int
        """
        nan_cnt = 0
        for value in values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt


