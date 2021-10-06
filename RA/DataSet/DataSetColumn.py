from prettytable import PrettyTable
from RA.DataSet.DataSetColumnNumStat import *
from RA.DataSet.DataSetColumnStrStat import *


class DataSetColumn:
    def __init__(self,
                 column_name: str,
                 values: list = None,
                 categorical: int = 25,
                 extended: bool = None,
                 rounding_factor: int = 2):
        self.__column_name = column_name
        self.__values = None
        self.__count = None
        self.__count_unique = None
        self.__nan_count = None
        self.__field_type = None
        self.__field_dtype = None
        self.__num_stat = None
        self.__str_stat = None
        self.__use_extended = extended
        self.__rounding_factor = rounding_factor
        self.__is_num_stat = False  # Отвечает за наличие числовых расчётов
        self.__is_str_stat = False
        if values is not None:
            self.fill_dataset_column(values=values,
                                     categorical=categorical,
                                     extended=extended)

    def __str__(self):
        table = PrettyTable()
        is_dataset = True if self.__values is not None and len(self.__values) > 0 else False
        table.title = f"{'Empty ' if not is_dataset else ''}Column \"{self.__column_name}\""
        table.field_names = ["Indicator", "Value"]
        table.add_row(["Сolumn name", self.get_column_name()])
        table.add_row(["Type", self.get_type()])
        table.add_row(["DType", self.get_dtype(threshold=0.15)])
        table.add_row(["Count", self.get_count()])
        table.add_row(["Count unique", self.get_count_unique()])
        table.add_row(["NaN count", self.get_nan_count()])

        if self.__is_num_stat:
            table.add_row(["Numerical indicators", "".join(len("Numerical indicators") * [" "])])
            table.add_row(["Min val", self.get_min()])
            table.add_row(["Mean val", self.get_mean()])
            table.add_row(["Median val", self.get_median()])
            table.add_row(["Max val", self.get_max()])

            if self.get_num_stat().get_normal_distribution():
                table.add_row(["Normal Distribution", "".join(len("Normal Distribution") * [" "])])
                table.add_row(["Mathematical mode", self.get_math_mode()])
                table.add_row(["Mathematical expectation", self.get_math_expectation()])
                table.add_row(["Mathematical dispersion", self.get_math_dispersion()])
                table.add_row(["Mathematical sigma", self.get_math_sigma()])
                table.add_row(["Coefficient of Variation", self.get_coef_of_variation()])
        return str(table)

    def fill_dataset_column(self,
                            values: list or pd.DataFrame,
                            categorical: int = 25,
                            extended: bool = None):
        """
        This method fill this class when we use values
        :param categorical:
        :param values: list of column values
        :param extended: The switch responsible for calculating the indicators of the normal distribution
        """
        self.__values = values
        self.__count = len(self.__values)
        self.__count_unique = len(list(set(self.__values)))
        self.__field_type = self.get_column_type()
        self.__field_dtype = "variable" if self.__count_unique >= categorical else "categorical"
        self.__nan_count = self.__get_nan_count()
        extended = extended if extended is not None else False
        if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
            self.__num_stat = NumericalIndicators(extended=extended)
            self.__num_stat.set_values(values=self.__values,
                                       extended=extended)
            self.__is_num_stat = True
        elif self.__field_type.startswith("str"):
            self.__str_stat = StringIndicators(extended=extended)
            self.__str_stat.set_values(values=self.__values,
                                       extended=extended)
            self.__is_str_stat = True

    def get_values(self):
        return self.__values

    def get_num_stat(self) -> NumericalIndicators:
        return self.__num_stat

    def get_str_stat(self) -> StringIndicators:
        return self.__str_stat

    def set_column_name(self, new_column_name: str) -> None:
        """
        This method sets the new name for current column
        :param new_column_name: The new name of this column
        :return: None
        """
        self.__column_name = new_column_name

    def get_column_name(self) -> str:
        """
        This method return the name of current column
        :return: Name of column
        """
        return self.__column_name

    def get_count(self) -> int:
        """
        This method returns count of values in this column
        :return: Count of values
        """
        if self.__count is None:
            raise Exception("The values were not loaded!")
        return self.__count

    def get_count_unique(self) -> int:
        """
        This method returns count of unique values in this column
        :return: Count of unique values
        """
        if self.__count_unique is None:
            raise Exception("The values were not loaded!")
        return self.__count_unique

    def get_nan_count(self) -> int:
        """
        This method returns count of NaN values in this column
        :return: Count of NaN values
        """
        if self.__nan_count is None:
            raise Exception("The values were not loaded!")
        return self.__nan_count

    def get_type(self) -> str:
        """
        This method returns type of column
        :return: Type of column
        """
        if self.__field_type is None:
            raise Exception("The values were not loaded!")
        return self.__field_type

    def get_dtype(self, threshold: float):
        if self.__field_dtype is None:
            raise Exception("The values were not loaded!")
        self.__field_dtype = "variable" if self.__count_unique >= len(self.__values) * threshold else "categorical"
        return self.__field_dtype

    def get_min(self) -> int or float or bool:
        """
        This method return minimal value of column
        :return Minimal value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_min()
        elif self.__is_str_stat:
            return self.__str_stat.get_min()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_max(self) -> int or float or bool:
        """
        This method return maximal value of column
        :return Maximal value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_max()
        elif self.__is_str_stat:
            return self.__str_stat.get_max()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_mean(self) -> int or float or bool:
        """
        This method return maximal value of column
        :return Mean value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_mean()
        elif self.__is_str_stat:
            return self.__str_stat.get_mean()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_median(self) -> int or float or bool:
        """
        This method return maximal value of column
        :return Median value of column
        """
        if self.__is_num_stat:
            return self.__num_stat.get_median()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_distribution(self):
        """
        This method return mathematical distribution dict
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_math_distribution()
        if self.__is_str_stat and self.get_str_stat().get_is_letter_counter():
            return self.__str_stat.get_letter_counter().get_letter_distribution()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_math_mode(self):
        """
        This method return mathematical mode
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_math_mode()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_math_expectation(self):
        """
        This method return mathematical expectation
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_math_expectation()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_math_dispersion(self):
        """
        This method return mathematical dispersion
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_math_dispersion()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_math_sigma(self):
        """
        This method return mathematical sigma
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_math_sigma()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_coef_of_variation(self):
        """
        This method return mathematical sigma
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_coef_of_variation()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_Z_score(self):
        """
        This method return mathematical sigma
        """
        if self.__is_num_stat and self.get_num_stat().get_is_normal_distribution():
            return self.__num_stat.get_normal_distribution().get_Z_score()
        else:
            raise Exception(f"The values in the column '{self.__column_name}' are not numbers!")

    def get_is_num_stat(self) -> bool:
        return self.__is_num_stat

    def get_is_extended(self) -> bool:
        return self.__use_extended

    def get_column_type(self) -> str:
        types = []
        for i in range(len(self.__values)):
            types.append(type(self.__values[i]).__name__)
        types = list(set(types))
        if len(types) == 1:
            return types[0]
        else:
            if len(types) == 2 and 'str' in types:
                return 'str'
            return "object"

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
            self.__num_stat: NumericalIndicators = NumericalIndicators(extended=False)
            self.__num_stat.get_from_json(data=data["Numerical indicators"])
            self.__is_num_stat = True
        if "String Indicators" in data:
            self.__str_stat: StringIndicators = StringIndicators(extended=False)
            self.__str_stat.get_from_json(data=data["String Indicators"])
            self.__is_str_stat = True

    def to_json(self) -> Dict[str, dict]:
        """
        This method export DataSet statistics as json
        :return: Dict[str, json]
        """
        if self.__values is not None:
            data = {"column_name": self.__column_name,
                    "count": self.__count,
                    "count_unique": self.__count_unique,
                    "count_NaN": self.__nan_count,
                    "type": self.__field_type,
                    "dtype": self.__field_dtype}
            if self.__field_type.startswith("int") or self.__field_type.startswith("float"):
                if not self.__num_stat.get_is_numerical_indicators():
                    self.__num_stat = NumericalIndicators(extended=self.__use_extended)
                    self.__num_stat.set_values(values=self.__values,
                                               extended=self.__use_extended)
                    self.__is_num_stat = True
                data["Numerical indicators"] = self.__num_stat.to_json()
            elif self.__field_type.startswith("str"):
                if not self.__str_stat.get_is_string_indicators():
                    self.__str_stat = StringIndicators(extended=self.__use_extended)
                    self.__str_stat.set_values(values=self.__values,
                                               extended=self.__use_extended)
                    self.__is_str_stat = True
                data["String Indicators"] = self.__str_stat.to_json()
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


