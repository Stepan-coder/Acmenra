import pandas as pd
from prettytable import PrettyTable


class CorrelationMatrix:
    def __init__(self, keys: list) -> None:
        """
        This method init the work of this class
        :param keys: List of DataSet column names
        """
        self.__keys = keys
        self.__correlation_matrix = {}
        for outkey in keys:
            self.__correlation_matrix[outkey] = {}
            for inkey in keys:
                self.__correlation_matrix[outkey][inkey] = float('nan')

    def __str__(self) -> str:
        table = PrettyTable()
        table.title = f"Correlation Matrix"
        table.field_names = ["Name"] + list(self.__keys)
        for key in self.__correlation_matrix:
            table.add_row([key] + list(self.__correlation_matrix[key].values()))
        return str(table)

    def is_cell_free(self, column_name_a: str, column_name_b: str) -> bool:
        """
        This method checks, is cell with coordinates (column_name_a, column_name_b) is not filled.
        :param column_name_a: column name of DataSet
        :param column_name_b: column name of DataSet
        :return: is cell filled
        """
        if column_name_a not in self.__keys:
            raise Exception(f"The \"{column_name_a}\" column does not exist in this dataset!")
        if column_name_b not in self.__keys:
            raise Exception(f"The \"{column_name_b}\" column does not exist in this dataset!")
        return pd.isna(self.__correlation_matrix[column_name_a][column_name_b])

    def add_corr(self, column_name_a: str, column_name_b: str, value: float) -> None:
        """
        This method fills the cell of correlation matrix
        :param column_name_a: column name of DataSet
        :param column_name_b: column name of DataSet
        :param value: coefficient of correlation between column_name_a values and column_name_b values
        :return: None
        """
        if column_name_a not in self.__keys:
            raise Exception(f"The \"{column_name_a}\" column does not exist in this dataset!")
        if column_name_b not in self.__keys:
            raise Exception(f"The \"{column_name_b}\" column does not exist in this dataset!")
        self.__correlation_matrix[column_name_a][column_name_b] = value
        self.__correlation_matrix[column_name_b][column_name_a] = value

    def get_corr(self, column_name_a: str, column_name_b: str) -> float:
        """
        This method returns the coefficient of correlation
        :param column_name_a: column name of DataSet
        :param column_name_b: column name of DataSet
        :return: coefficient of correlation between column_name_a values and column_name_b values
        """
        if column_name_a not in self.__keys:
            raise Exception(f"The \"{column_name_a}\" column does not exist in this dataset!")
        if column_name_b not in self.__keys:
            raise Exception(f"The \"{column_name_b}\" column does not exist in this dataset!")
        return self.__correlation_matrix[column_name_a][column_name_b]
