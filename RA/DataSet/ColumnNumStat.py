from prettytable import PrettyTable
from RA.DataSet.ColumnType import *
from RA.DataSet.ColumnNumStatExt import *


class ColumnNumStat:
    def __init__(self,
                 column_name: str,
                 values: list,
                 extended: bool,
                 categorical: float = 0.15) -> None:
        self.__column_name = column_name
        self.__is_extended = extended
        self.__count = len(values)  # Указываем явно, потому что этот класс не должен хранить все значения с колонки
        self.__count_unique = len(list(set(values)))
        self.__field_type = self.__get_column_type(values=values)
        self.__field_dtype = "variable" if self.__count_unique >= self.__count * categorical else "categorical"
        self.__nan_count = ColumnNumStat.__get_nan_count(values=values)
        self.__num_stat = NumericalIndicators(values=values, extended=extended)

    def __str__(self):
        table = PrettyTable()
        table.title = f"{'Simple' if not self.__is_extended else 'Extended'} Column \"{self.__column_name}\""
        table.field_names = ["Indicator", "Value"]
        table.add_row(["Сolumn name", self.column_name])
        table.add_row(["Type", self.type])
        table.add_row(["DType", self.dtype])
        table.add_row(["Count", self.count])
        table.add_row(["Count unique", self.unique_count])
        table.add_row(["NaN count", self.nan_count])
        table.add_row(["Numerical indicators", "".join(len("Numerical indicators") * [" "])])
        table.add_row(["Min val", self.min()])
        table.add_row(["Mean val", self.mean()])
        table.add_row(["Median val", self.median()])
        table.add_row(["Max val", self.max()])

        if self.get_num_stat().get_values_distribution():
            table.add_row(["Normal Distribution", "".join(len("Normal Distribution") * [" "])])
            table.add_row(["Mathematical mode", self.get_math_mode()])
            table.add_row(["Mathematical expectation", self.get_math_expectation()])
            table.add_row(["Mathematical dispersion", self.get_math_dispersion()])
            table.add_row(["Mathematical sigma", self.get_math_sigma()])
            table.add_row(["Coefficient of Variation", self.get_coef_of_variation()])
        return str(table)

    def __len__(self) -> int:
        """
        This method returns the len[count] of values in this column
        """
        return self.__count

    @property
    def column_name(self) -> str:
        """
        This method return the name of current column
        """
        return self.__column_name

    @property
    def type(self) -> ColumnType:
        """
        This method returns type of column
        """
        if self.__field_type is None:
            raise Exception("The values were not loaded!")
        return self.__field_type

    @property
    def dtype(self):
        """
        This method returns the real type of column
        """
        return self.__field_dtype

    @property
    def count(self) -> int:
        """"
        This method returns count of values in this column
        """
        return self.__count

    @property
    def unique_count(self) -> int:
        """
        This method returns count of unique values in this column
        """
        return self.__count_unique

    @property
    def nan_count(self) -> int:
        """
        This method returns count of NaN values in this column
        """
        return self.__nan_count

    @property
    def is_extended(self) -> bool:
        return self.__is_extended

    def get_num_stat(self) -> NumericalIndicators:
        return self.__num_stat

    def min(self) -> int or float:
        """
        This method return minimal value of column
        """
        return self.__num_stat.get_min()

    def max(self) -> int or float:
        """
        This method return maximal value of column
        """
        return self.__num_stat.get_max()

    def mean(self) -> int or float:
        """
        This method return maximal value of column
        """
        return self.__num_stat.get_mean()

    def median(self) -> int or float:
        """
        This method return maximal value of column
        """
        return self.__num_stat.get_median()

    def get_values_distribution(self) -> Dict[float or int, float]:
        """
        This method returns the percentage of values in the column
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_distribution()

    def get_math_mode(self) -> int or float:
        """
        This method return mathematical mode
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_math_mode()

    def get_math_expectation(self) -> int or float:
        """
        This method return mathematical expectation
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_math_expectation()

    def get_math_dispersion(self) -> float:
        """
        This method return mathematical dispersion
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_math_dispersion()

    def get_math_sigma(self) -> float:
        """
        This method return mathematical sigma
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_math_sigma()

    def get_coef_of_variation(self) -> float:
        """
        This method return mathematical sigma
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_coef_of_variation()

    def get_Z_score(self) -> float:
        """
        This method return mathematical sigma
        """
        if not self.get_num_stat().get_is_extended():
            raise Exception(f"Statistics have not been calculated for column '{self.__column_name}' yet! "
                            f"To get statistical values, use 'get_column_statinfo' with the 'extended' parameter")
        return self.__num_stat.get_values_distribution().get_Z_score()

    def __get_column_type(self, values: list) -> ColumnType:
        """
        This method learns the column type
        """
        types = []
        for i in range(len(values)):
            column_type = type(values[i]).__name__
            if column_type == 'bool':
                types.append(ColumnType.BOOLEAN)
            elif column_type == 'int':
                types.append(ColumnType.INTEGER)
            elif column_type == 'float':
                types.append(ColumnType.FLOAT)
            elif column_type == 'str':
                types.append(ColumnType.STRING)
        if len(list(set(types))) == 1:
            return types[0]
        else:
            if len(types) == 2 and ColumnType.STRING in list(set(types)):
                return ColumnType.STRING
            return ColumnType.OBJECT

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

    @staticmethod
    def __get_nan_count(values: list) -> int:
        """
        This method calculate count of NaN values
        """
        nan_cnt = 0
        for value in values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt


