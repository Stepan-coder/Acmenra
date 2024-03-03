import os
import sys
import json
import math
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from tqdm import tqdm
from typing import Any, List, Dict
from scipy import stats
from prettytable import PrettyTable
from RA.DataSet.ColumnStr import *
from RA.DataSet.ColumnNum import *
from RA.DataSet.ColumnType import *
from RA.DataSet.DataSetStat import *
from RA.DataSet.DataSetStatus import *
from RA.DataSet.ColumnNumStat import *
from RA.DataSet.ColumnStrStat import *


class DataSet(object):
    def __init__(self, dataset_name: str, show: bool = False) -> None:
        """
        This is an init method
        :param dataset_name: User nickname of dataset
        :param show: Do I need to show what is happening
        """
        self.__dataset_name = dataset_name
        self.__delimiter = ","
        self.__encoding = 'utf-8'
        self.__is_dataset_loaded = False
        self.__dataset = None  # Dataset in pd.DataFrame
        self.__dataset_len = None  # Number of records in the dataset
        self.__dataset_columns_name = None  # Column names in the dataset
        self.__dataset_columns_name_count = None  # Number of columns in the dataset
        self.__dataset_file = None
        self.__dataset_save_path = None
        self.__dataset_analytics = {}

    def __str__(self):
        table = PrettyTable()
        is_dataset = True if self.__dataset is not None and self.__dataset_len > 0 else False
        table.title = f"{'Empty ' if not is_dataset else ''}DataSet \"{self.__dataset_name}\""
        if is_dataset:
            table.title += f" ({len(self.__dataset_columns_name)} "
            table.title += "columns)" if self.__dataset_len > 1 else "column)"
        table.field_names = ["Column name", "Type", "Data type", "Count", "Count unique", "NaN count"]
        if is_dataset:
            for key in self.__dataset_columns_name:
                column = self.get_column_stat(column_name=key, extended=False)
                if column.dtype == 'variable':
                    dtype = "\033[32m {}\033[0m".format(column.dtype)
                else:
                    dtype = "\033[31m {}\033[0m".format(column.dtype)
                table.add_row([column.column_name,
                               column.type,
                               dtype,
                               column.count,
                               column.unique_count,
                               column.nan_count])
        return str(table)

    def __iter__(self) -> Dict[str, Any]:
        """
        This method allows you to iterate over a data set in a loop. I.e. makes it iterative
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        for i in range(len(self.__dataset)):
            yield self.get_row(index=i)

    def __reversed__(self):
        """
        This method return a reversed copy of self-class
        """
        copied_class = copy.copy(self)
        copied_class.reverse()
        return copied_class

    def __instancecheck__(self, instance: Any) -> bool:
        """
        This method checks is instance type is DataSet
        :param instance: Checked value
        """
        return isinstance(instance, type(self))

    def __len__(self) -> int:
        """
        This method returns count rows in this dataset
        """
        return len(self.__dataset) if self.__dataset is not None else 0

    @property
    def name(self) -> str:
        """
        This method returns the dataset name of the current DataSet
        """
        return self.__dataset_name

    @property
    def status(self) -> DataSetStatus:
        if not self.is_loaded and len(self) == 0:
            return DataSetStatus.CREATED
        elif self.is_loaded and len(self) == 0:
            return DataSetStatus.EMPTY
        elif self.is_loaded and len(self) > 0:
            return DataSetStatus.ENABLE

    @property
    def is_loaded(self) -> bool:
        """
        This property returns the state of this DataSet
        """
        return self.__is_dataset_loaded

    @property
    def delimiter(self) -> str:
        """
        This property returns a delimiter character
        """
        return self.__delimiter

    @property
    def encoding(self) -> str:
        """
        This property returns the encoding of the current dataset file
        :return: Encoding for the dataset. Example: 'utf-8', 'windows1232'
        """
        return self.__encoding

    @property
    def columns_name(self) -> List[str]:
        """
        This property return column names of dataset pd.DataFrame
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        return list(self.__dataset_columns_name)

    @property
    def columns_count(self) -> int:
        """
        This method return count of column names of dataset pd.DataFrame
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet")
        return self.__dataset_columns_name_count

    @property
    def supported_formats(self) -> List[str]:
        """
        This property returns a list of supported files
        """
        return [".xls", ".xlsx", ".xlsm", ".xlt", ".xltx", ".xlsb", '.ots', '.ods']

    def head(self, n: int = 5, full_view: bool = False) -> str:
        """
        This method prints the first n rows
        :param full_view:
        :param n: Count of lines
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            n = len(self.__dataset)
        table = PrettyTable()
        table.title = self.__dataset_name
        table.field_names = self.__dataset_columns_name
        for i in range(n):
            this_row = self.get_row(index=i)
            table_row = []
            for column_name in self.__dataset_columns_name:
                column_type = self.get_column_stat(column_name=column_name, extended=False).type
                if not full_view and column_type == ColumnType.STRING and isinstance(this_row[column_name], str) and \
                        len(this_row[column_name]) > 50:
                    table_row.append(f"{str(this_row[column_name])[:47]}...")
                else:
                    table_row.append(this_row[column_name])
            table.add_row(table_row)
        return str(table)

    def tail(self, n: int = 5, full_view: bool = False) -> None:
        """
        This method prints the last n rows
        :param full_view:
        :param n: Count of lines
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            n = len(self.__dataset)
        table = PrettyTable()
        table.title = self.__dataset_name
        table.field_names = self.__dataset_columns_name
        for i in range(self.__dataset_len - n, self.__dataset_len):
            this_row = self.get_row(index=i)
            table_row = []
            for column_name in self.__dataset_columns_name:
                column_type = self.get_column_stat(column_name=column_name, extended=False).get_type()
                if not full_view and column_type.startswith("str") and isinstance(this_row[column_name], str) and \
                        len(this_row[column_name]) > 50:
                    table_row.append(f"{str(this_row[column_name])[:47]}...")
                else:
                    table_row.append(this_row[column_name])
            table.add_row(table_row)
        print(table)

    def set_name(self, dataset_name: str) -> None:
        """
        This method sets the project_name of the DataSet
        :param dataset_name: Name of this
        """
        if not isinstance(dataset_name, str):
            raise Exception("The name must be a string!")
        if not len(dataset_name) > 0:
            raise Exception("The name must be longer than 0 characters!")
        self.__dataset_name = dataset_name

    def set_saving_path(self, path: str) -> None:
        """
        This method removes the column from the dataset
        :param path: The path to save the "DataSet" project
        """
        self.__dataset_save_path = path

    def set_delimiter(self, delimiter: str) -> None:
        """
        This method sets the delimiter character
        :param delimiter: Symbol-split in a .csv file
        """
        if not isinstance(delimiter, str):
            raise Exception("The delimiter must be a one-symbol string!")
        if len(delimiter) > 1 or len(delimiter) == 0:
            raise Exception("A separator with a length of 1 character is allowed!")
        self.__delimiter = delimiter

    def set_encoding(self, encoding: str) -> None:
        """
        This method sets the encoding for the future export of the dataset
        :param encoding: Encoding for the dataset. Example: 'utf-8', 'windows1232'
        """
        if not isinstance(encoding, str):
            raise Exception("The encoding must be a string!")
        if len(encoding) == 0:
            raise Exception("The name must be longer than 0 characters!")
        self.__encoding = encoding

    def set_to_field(self, column: str, index: int, value: Any) -> None:
        """
        This method gets the value from the dataset cell
        :param column: The name of the dataset column
        :param index: Index of the dataset string
        :param value: The value that we want to write
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        if column not in self.__dataset_columns_name:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        if column in self.__dataset_analytics:
            self.__dataset_analytics.pop(column)
        self.__dataset.loc[index, column] = value

    def get_from_field(self, column: str, index: int) -> Any:
        """
        This method gets the value from the dataset cell
        :param column: The name of the dataset column
        :param index: Index of the dataset string
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index >= self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        if column not in self.__dataset_columns_name:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        return self.__dataset.at[index, column]

    def add_row(self, new_row: Dict[str, Any]) -> None:
        """
        This method adds a new row to the dataset
        :param new_row: The string to be added to the dataset. In dictionary format,
        where the key is the column name and the value is a list of values
        """
        if not self.__is_dataset_loaded:
            warnings.warn(f'The dataset was not loaded. '
                          f'An empty dataset was created with the columns {list(new_row.keys())}!', UserWarning)
            self.create_empty_dataset(columns_names=list(new_row.keys()))
        if len(set(new_row.keys())) != len(new_row.keys()):
            raise Exception(f"Column names should not be repeated!")
        for column in new_row:
            if column not in self.__dataset_columns_name:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        for column in self.__dataset_columns_name:
            if column not in new_row.keys():
                raise Exception(f"The \"{column}\" column is missing!")
        self.__dataset.loc[len(self.__dataset)] = [new_row[d] for d in self.__dataset_columns_name]
        for key in self.__dataset_columns_name:
            if key in self.__dataset_analytics:
                self.__dataset_analytics.pop(key)
        self.__dataset_len += 1

    def get_row(self, index: int) -> Dict[str, Any]:
        """
        This method returns a row of the dataset in dictionary format, where the keys are the column names and the
        values are the values in the columns
        :param index: Index of the dataset string
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if index < 0:
            raise Exception("The row index must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row index must be less than the number of rows in the dataset!")
        result = {}
        for column in self.__dataset_columns_name:
            if column not in result:
                result[column] = self.get_from_field(column=column,
                                                     index=index)
        return result

    def delete_row(self, index: int) -> None:
        """
        This method delete row from dataset
        :param index: Index of the dataset string
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if index < 0:
            raise Exception("The row index must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        self.__dataset = self.__dataset.drop(index=index).reset_index(level=0, drop=True)
        for key in self.__dataset_columns_name:
            if key in self.__dataset_analytics:
                self.__dataset_analytics.pop(key)
        self.__dataset_len = self.__dataset_len - 1 if self.__dataset_len > 0 else 0

    def Column(self, column_name: str) -> ColumnStr or ColumnNum:
        """
        This method summarizes the values from the columns of the dataset and returns them as a list of tuples
        :param column_name: Name of DataSet column
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        col_type = self.get_column_stat(column_name=column_name, extended=False).type
        if col_type == ColumnType.STRING:
            return ColumnStr(column=self.__dataset[column_name], column_type=col_type)
        elif col_type == ColumnType.INTEGER or col_type == ColumnType.FLOAT:
            return ColumnNum(column=self.__dataset[column_name], column_type=col_type)

    def add_column(self, column_name: str, values: list, dif_len: bool = False) -> None:
        """
        This method adds the column to the dataset on the right
        :param column_name: String name of the column to be added
        :param values: List of column values
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        """
        if not self.__is_dataset_loaded:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if column_name in self.__dataset_columns_name:
            raise Exception(f"The '{column_name}' column already exists in the presented dataset!")
        if len(self.__dataset) != len(values) and len(self.__dataset) != 0:
            if not dif_len:
                raise Exception("The column and dataset must have the same size!")
        self.__dataset[column_name] = values
        self.__dataset_columns_name = self.__dataset.keys()
        self.__dataset_columns_name_count += 1

    def get_column(self, column_name: str) -> list:
        """
        This method summarizes the values from the columns of the dataset and returns them as a list of tuples
        :param column_name: Name of DataSet column
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        return self.__dataset[column_name].to_list()

    def rename_column(self, column_name: str, new_column_name: str) -> None:
        """
        This method renames the column in the dataset
        :param column_name: The name of the column that we are renaming
        :param new_column_name: New name for the "column" column
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        if new_column_name in self.__dataset_columns_name:
            raise Exception(f"The \"{new_column_name}\" column does already exist in this dataset!")
        self.__dataset = self.__dataset.rename(columns={column_name: new_column_name})
        if column_name in self.__dataset_analytics:
            column_analytic = self.__dataset_analytics[column_name]
            self.__dataset_analytics.pop(column_name)
            self.__dataset_analytics[new_column_name] = column_analytic
            self.__dataset_analytics[new_column_name].set_column_name(new_column_name=new_column_name)
        self.__dataset_columns_name = self.__dataset.keys()

    def delete_column(self, column_name: str) -> None:
        """
        This method removes the column from the dataset
        :param column_name: List of column names
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        self.__dataset = self.__dataset.drop([column_name], axis=1)
        if column_name in self.__dataset_analytics:
            self.__dataset_analytics.pop(column_name)
        self.__dataset_columns_name = self.__dataset.keys()
        self.__dataset_columns_name_count = len(self.__dataset.keys())

    def set_columns_types(self, new_column_types: type, exception: Dict[str, type] = None) -> None:
        """
        This method converts column types
        :param new_column_types: New type of dataset columns (excluding exceptions)
        :param exception: Fields that have a different type from the main dataset type
        """
        for field in self.__dataset:
            if exception is None:
                self.set_column_type(column_name=field, new_column_type=new_column_types)
            else:
                if field in exception:
                    self.set_column_type(column_name=field, new_column_type=exception[field])
                else:
                    self.set_column_type(column_name=field, new_column_type=new_column_types)

    def set_column_type(self, column_name: str, new_column_type: type) -> None:
        """
        This method converts column type
        :param column_name: The name of the column in which we want to change the type
        :param new_column_type: Field type
        """
        if new_column_type != str and new_column_type != int and new_column_type != float and new_column_type != bool:
            raise Exception(f"'{new_column_type}' is an invalid data type for conversion. "
                            f"Valid types: bool, int, float, str")
        if column_name in self.__dataset:
            primary_type = str(self.__dataset[column_name].dtype)
            if new_column_type == float or new_column_type == int:
                self.__dataset[column_name] = self.__dataset[column_name].replace(",", ".", regex=True) \
                    .replace(" ", "", regex=True) \
                    .fillna(0)
                try:
                    self.__dataset[column_name] = self.__dataset[column_name].astype(float)
                except Exception as e:
                    raise Exception(str(e).capitalize())
                if new_column_type == int:
                    try:
                        self.__dataset[column_name] = self.__dataset[column_name].astype(int)
                    except Exception as e:
                        raise Exception(str(e).capitalize())
            else:
                try:
                    self.__dataset[column_name] = self.__dataset[column_name].astype(new_column_type)
                except Exception as e:
                    raise Exception(str(e).capitalize())
            secondary_type = str(self.__dataset[column_name].dtype)
            print(f"Convert DataSet field \'{column_name}\': {primary_type} -> {secondary_type}")
        else:
            raise Exception("There is no such column in the presented dataset!")

    def get_column_stat(self, column_name: str, extended: bool) -> ColumnNumStat or ColumnStrStat:
        """
        This method returns statistical analytics for a given column
        :param column_name: The name of the dataset column for which we output statistics
        :param extended: Responsible for calculating additional parameters
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        col = column_name
        if col in self.__dataset_analytics:
            if extended and not self.__dataset_analytics[col].get_is_extended():
                if self.__get_column_type(col) == ColumnType.INTEGER or self.__get_column_type(col) == ColumnType.FLOAT:
                    self.__dataset_analytics[col] = ColumnNumStat(col, list(self.__dataset[col]), extended)
                elif self.__get_column_type(col) == ColumnType.STRING:
                    self.__dataset_analytics[col] = ColumnStrStat(col, list(self.__dataset[col]), extended)
        else:
            if self.__get_column_type(col) == ColumnType.INTEGER or self.__get_column_type(col) == ColumnType.FLOAT:
                self.__dataset_analytics[col] = ColumnNumStat(col, list(self.__dataset[col]), extended)
            elif self.__get_column_type(col) == ColumnType.STRING:
                self.__dataset_analytics[col] = ColumnStrStat(col, list(self.__dataset[col]), extended)
        return self.__dataset_analytics[col]

    def get_columns_stat(self, extended: bool) -> Dict[str, ColumnNumStat]:
        """
        This method returns DataSet columns stat info
        """
        for key in self.__dataset_columns_name:
            if key in self.__dataset_analytics:
                if not self.__dataset_analytics[key].get_is_extended() and extended:
                    self.get_column_stat(key, extended)
            else:
                self.get_column_stat(key, extended)
        return self.__dataset_analytics

    def reverse(self) -> None:
        """
        This method expands the order of rows in the dataset
        """
        self.__dataset = self.__dataset.reindex(index=self.__dataset.index[::-1])
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)

    def fillna(self) -> None:
        """
        This method automatically fills in "null" values:
         For "int" -> 0.
         For "float" -> 0.0.
         For "str" -> "-".
        """
        for key in self.__dataset_columns_name:
            column_type = self.get_column_stat(column_name=key, extended=False).get_type()
            if column_type.startswith('str'):
                self.__dataset[key] = self.__dataset[key].fillna(value="⁣")
            elif column_type.startswith('int') or column_type.startswith('float'):
                self.__dataset[key] = self.__dataset[key].fillna(value=0)
        self.update_dataset_info()

    def equals(self, dataset) -> bool:
        diff_responce = self.diff(dataset=dataset)
        if 'length' not in diff_responce and 'columns' not in diff_responce and len(diff_responce["rows"].keys()) == 0:
            return True
        return False

    def diff(self, dataset) -> json:
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        some: DataSet = dataset
        report = {}
        if self.__dataset_len != some.__dataset_len:
            report['length'] = {self.name: f"{self.__dataset_len}", some.name: f"{some.__dataset_len}"}
        if some.columns_name != self.columns_name:
            not_exist = [column for column in some.columns_name if column not in self.__dataset_columns_name]
            missing = [column for column in self.__dataset_columns_name if column not in some.columns_name]
            report['columns'] = list(set(not_exist + missing))
        report['rows'] = {}
        for column in self.__dataset_columns_name:
            if column in self.__dataset_columns_name and column in some.columns_name:
                indexes = DataSet.__dif_lists_index(list_a=self.Column(column_name=column).values(),
                                                    list_b=some.Column(column_name=column).values())
                for index in indexes:
                    if index not in list(report['rows'].keys()):
                        report['rows'][index] = [column]
                    else:
                        report['rows'][index].append(column)
        print(len(report["rows"].keys()))
        return report

    def split(self, count: int) -> List:
        """
        This method automatically divides the DataSet into a list of DataSets with a maximum of "count" rows in each
        :param count: Maximal count of rows in new DataSets
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if count <= 0:
            raise Exception("Count of rows 'count' should be large, then 0!")
        counter = 0
        result = []
        while counter * count < len(self.__dataset):
            this_dataset = DataSet(dataset_name=f"splited_{self.__dataset_name}_{counter}")
            this_dataset.load_DataFrame(dataframe=self.__dataset[counter * count: (counter + 1) * count])
            this_dataset.set_delimiter(delimiter=self.delimiter)
            this_dataset.set_encoding(encoding=self.encoding)
            result.append(this_dataset)
            counter += 1
        return result

    def sort_by_column(self, column_name: str, reverse: bool = False) -> None:
        """
        This method sorts the dataset by column "column_name"
        :param column_name: The name of the column by which the sorting will take place
        :param reverse: The parameter responsible for the sorting order (ascending/descending)
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_columns_name:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        column_type = self.get_column_stat(column_name=column_name, extended=False).get_type()
        self.fillna()
        for sort_type in ["mergesort", "quicksort", "heapsort", "stable"]:
            try:
                if column_type.startswith("str"):
                    some = {}
                    for i in range(len(self.__dataset)):
                        some[i] = self.__dataset.at[i, column_name]
                    some = list(dict(sorted(some.items(), key=lambda x: len(x[1]))).keys())
                    self.__dataset = self.__dataset.reindex(some)
                else:
                    self.__dataset = self.__dataset.sort_values(by=column_name,
                                                                ascending=reverse,
                                                                kind=sort_type)
                self.__dataset = self.__dataset.reset_index(level=0, drop=True)
                return
            except:
                pass

    def get_correlations(self) -> CorrelationMatrix:
        """
        This method calculate correlations between columns
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        corr_matrix = CorrelationMatrix(keys=self.__dataset_columns_name)
        for keya in self.__dataset_columns_name:
            row_type = self.get_column_stat(column_name=keya, extended=False).get_type()
            for keyb in self.__dataset_columns_name:
                if corr_matrix.is_cell_free(keya, keyb):
                    column_type = self.get_column_stat(column_name=keyb, extended=False).get_type()
                    if (column_type.startswith('int') or column_type.startswith('bool') or column_type.startswith(
                            'float')) \
                            and (
                            row_type.startswith('int') or row_type.startswith('bool') or row_type.startswith('float')):
                        corr_matrix.add_corr(keya, keyb, np.corrcoef(self.__dataset[keya], self.__dataset[keyb])[0][1])
                    else:
                        corr_matrix.add_corr(keya, keyb, float('nan'))
        return corr_matrix

    def get_DataFrame(self) -> pd.DataFrame:
        """
        This method return dataset as pd.DataFrame
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been uploaded yet!")
        return self.__dataset

    def join_DataFrame(self, dataframe: pd.DataFrame, dif_len: bool = False) -> None:
        """
        This method attaches a new dataset to the current one (at right)
        :param dataframe: The pd.DataFrame to be attached to the current one
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        """
        if self.__dataset is None:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if len(dataframe) == 0:
            raise Exception("You are trying to add an empty dataset")
        if len(self.__dataset) != len(dataframe) and len(self.__dataset) > 0:
            if not dif_len:
                raise Exception("The pd.DataFrames must have the same size!")
        columns_names = list(self.__dataset.keys()) + list(dataframe.keys())
        if len(set(columns_names)) != len(columns_names) and len(list(self.__dataset.keys())) != 0:
            raise Exception("The current dataset and the new dataset have the same column names!")
        self.__dataset = self.__dataset.join(dataframe,
                                             how='outer')
        self.__update_dataset_base_info()

    def join_DataSet(self, dataset, dif_len: bool = False) -> None:
        """
        This method attaches a new dataset to the current one(at right)
        :param dataset: The DataSet object to be attached to the current one
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        """
        if self.__dataset is None:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if len(dataset) == 0:
            raise Exception("You are trying to add an empty dataset")
        if len(self.__dataset) != len(dataset) and len(self.__dataset) > 0:
            if not dif_len:
                raise Exception("The pd.DataFrames must have the same size!")
        columns_names = list(self.__dataset.keys()) + list(dataset.get_keys())
        if len(set(columns_names)) != len(columns_names):
            raise Exception("The current dataset and the new dataset have the same column names!")
        self.__dataset = self.__dataset.join(dataset.get_DataFrame(),
                                             how='outer')
        self.__dataset_analytics = merge_two_dicts(self.__dataset_analytics, dataset.get_columns_stat_info())
        self.__update_dataset_base_info()

    def concat_DataFrame(self, dataframe: pd.DataFrame) -> None:
        """
        This method attaches a new dataset to the current one (at bottom)
        :param dataframe: The pd.DataFrame to be attached to the current one
        """
        if self.__dataset is None:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if len(dataframe) == 0:
            raise Exception("You are trying to add an empty dataset")
        columns_names = set(list(self.__dataset.keys()) + list(dataframe.keys()))
        if len(self.__dataset.keys()) != len(columns_names) and len(self.__dataset) > 0:
            raise Exception("The current dataset and the new dataset have the different column names!")
        self.__dataset = pd.concat([self.__dataset, dataframe])
        self.__dataset_analytics = {}
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)
        self.__update_dataset_base_info()

    def concat_DataSet(self, dataset) -> None:
        """
        This method attaches a new dataset to the current one (at bottom)
        :param dataset: The DataSet object to be attached to the current one
        """
        if self.__dataset is None:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if len(dataset) == 0:
            raise Exception("You are trying to add an empty dataset")
        columns_names = set(list(self.__dataset.keys()) + list(dataset.get_keys()))
        if len(self.__dataset.keys()) != len(columns_names) and len(self.__dataset) > 0:
            raise Exception("The current dataset and the new dataset have the different column names!")
        self.__dataset = pd.concat([self.__dataset, dataset.get_DataFrame()])
        self.__dataset_analytics = {}
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)
        self.__update_dataset_base_info()

    def update_dataset_info(self) -> None:
        """
        This method updates, the analitic-statistics data about already precalculated columns
        """
        self.__update_dataset_base_info()
        for key in self.__dataset_columns_name:
            is_extended = False
            if key in self.__dataset_analytics:
                is_extended = self.__dataset_analytics[key].get_is_extended()
            if self.__get_column_type(column_name=key) == ColumnType.INTEGER or \
                    self.__get_column_type(column_name=key) == ColumnType.FLOAT:
                self.__dataset_analytics[key] = ColumnNumStat(column_name=key,
                                                              values=self.__dataset[key],
                                                              extended=is_extended)
            elif self.__get_column_type(column_name=key) == ColumnType.STRING:
                self.__dataset_analytics[key] = ColumnStrStat(column_name=key,
                                                              values=self.__dataset[key],
                                                              extended=is_extended)

    def create_empty_dataset(self,
                             columns_names: list = None,
                             delimiter: str = ",",
                             encoding: str = 'utf-8'):
        """
        This method creates an empty dataset
        :param columns_names: List of column names
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if columns_names is not None:
            if len(set(columns_names)) != len(columns_names):
                raise Exception(f"Column names should not be repeated!")
            self.__dataset = pd.DataFrame(columns=columns_names)
        else:
            self.__dataset = pd.DataFrame()
        self.__dataset_len = 0
        self.__delimiter = delimiter
        self.__encoding = encoding
        self.__dataset_columns_name = self.__dataset.keys()
        self.__dataset_columns_name_count = len(self.__dataset.keys())
        self.__is_dataset_loaded = True
        self.__update_dataset_base_info()

    def create_dataset_from_list(self,
                                 data: List[List[Any]],
                                 columns: List[str],
                                 delimiter: str = ",",
                                 encoding: str = 'utf-8') -> None:
        """
        This method creates a dataset from list of columns values
        :param data: List[List[Any] - data columns]
        :param columns: List of column names
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if len(data) == 0 or len(columns) == 0:
            raise Exception('The number of columns in the table must be greater than 0!')
        if len(set(columns)) != len(columns):
            raise Exception('In the table, column names should not be duplicated!')
        if len(columns) != len(data):
            raise Exception('There should be as many column names as there are columns in "data"!')
        self.__dataset = pd.DataFrame({columns[i]: data[i] for i in range(len(columns))})
        self.set_delimiter(delimiter=delimiter)
        self.set_encoding(encoding=encoding)
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_DataFrame(self, dataframe: pd.DataFrame) -> None:
        """
        This method loads the dataset into the DataSet class
        :param dataframe: Explicitly specifying pd. DataFrame as a dataset
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        self.__dataset = dataframe
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_csv_dataset(self,
                         csv_file: str,
                         delimiter: str,
                         encoding: str = 'utf-8') -> None:
        """
        This method loads the dataset into the DataSet class
        :param csv_file: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if csv_file is not None:  # Checking that the uploaded file has the .csv format
            if not csv_file.endswith(".csv"):
                raise Exception("The dataset format should be '.csv'!")
        if csv_file is not None and delimiter is None:  # Checking that a separator is specified
            raise Exception("When loading a dataset from a .csv file, you must specify a separator character!")
        self.__delimiter = delimiter
        self.__dataset_file = csv_file
        self.__dataset = self.__read_from_csv(filename=str(csv_file),
                                              delimiter=delimiter,
                                              encoding=encoding)
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_excel_dataset(self,
                           excel_file: str,
                           sheet_name: str) -> None:
        """
        This method loads the dataset into the DataSet class
        :param sheet_name: Name of sheet in excel file
        :param excel_file: The name of the excel file
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if excel_file is not None:  # Checking that the uploaded file has the .csv format
            if not excel_file.endswith(tuple(DataSet.supported_formats)):
                raise Exception(f"The dataset format should be {', '.join(DataSet.supported_formats)}!")
        if sheet_name not in pd.ExcelFile(excel_file).sheet_names:
            raise Exception(f"Sheet name \'{sheet_name}\' not found!")
        self.__dataset_file = excel_file
        self.__dataset = self.__read_from_xlsx(filename=str(excel_file),
                                               sheet_name=sheet_name)
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_dataset_project(self,
                             dataset_project_folder: str,
                             json_config_filename: str) -> None:
        """
        This method loads the dataset into the DataSet class
        :param dataset_project_folder: The path to the .csv file
        :param json_config_filename:
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if not json_config_filename.endswith(".json"):
            raise Exception("The file must be in .json format!")
        if not os.path.exists(dataset_project_folder):
            raise Exception("The specified path was not found!")

        with open(f"{dataset_project_folder}\\{json_config_filename}", 'r') as json_file:
            dataset_info = json.load(json_file)
            json_file.close()
            self.__dataset_file = dataset_info["dataset_filename"]
            self.__delimiter = dataset_info["delimiter"]
            self.__encoding = dataset_info["encoding"]
            self.__dataset = self.__read_from_csv(filename=os.path.join(dataset_project_folder, self.__dataset_file),
                                                  delimiter=self.__delimiter,
                                                  encoding=self.__encoding)
            self.__read_dataset_info_from_json(dataset_info)
        self.__is_dataset_loaded = True

    def export(self,
               dataset_name: str = None,
               dataset_folder: str = None,
               including_json: bool = False,
               including_plots: bool = False,
               delimeter: str = None,
               encoding: str = None) -> None:
        """
        This method exports the dataset as DataSet Project
        :param delimeter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :param dataset_name: New dataset name (if the user wants to specify another one)
        :param dataset_folder: The folder to place the dataset files in
        :param including_json: Responsible for the export the .json config file together with the dataset
        :param including_plots: Responsible for the export the plots config file together with the dataset
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        print(f"Saving DataSet \'{self.__dataset_name}\'...")
        pass

        if dataset_name is not None:  # Если явно указано имя выходного файла
            dataset_filename = dataset_name
        elif self.__dataset_name is not None:  # Иначе - берём имя датасета, указанное в классе
            dataset_filename = self.__dataset_name
        else:  # Иначе - берём имя из загруженного файла
            dataset_filename = os.path.basename(self.__dataset_file).replace(".csv", "")

        folder = ""
        if self.__dataset_save_path is not None:
            if os.path.exists(self.__dataset_save_path):  # Если родительский путь верен
                folder = os.path.join(self.__dataset_save_path, dataset_filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
        elif dataset_folder is not None:  # Если явно указана папка, куда сохраняемся
            if os.path.exists(dataset_folder):  # Если родительский путь верен
                folder = os.path.join(dataset_folder, dataset_filename)
                if not os.path.exists(folder):
                    os.makedirs(folder)
            else:
                raise Exception("There is no such folder!")
        else:
            folder = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), dataset_filename)
            if not os.path.exists(folder):
                os.makedirs(folder)

        if encoding is not None and isinstance(encoding, str):
            self.set_encoding(encoding=encoding)
        if delimeter is not None and isinstance(delimeter, str):
            self.set_delimiter(delimiter=delimeter)

        if including_json and self.__dataset is not None:
            json_config = {"dataset_filename": f"{dataset_filename}.csv",
                           "columns_names": list(self.__dataset_columns_name),
                           "columns_count": self.__dataset_columns_name_count,
                           "rows": self.__dataset_len,
                           "delimiter": self.__delimiter,
                           "encoding": self.__encoding,
                           "columns": {}}
            for key in self.__dataset_columns_name:
                if key not in self.__dataset_analytics:
                    self.__dataset_analytics[key] = self.get_column_stat(column_name=key, extended=True)
                json_config["columns"] = merge_two_dicts(json_config["columns"],
                                                         self.__dataset_analytics[key].to_json())
            with open(os.path.join(folder, f"{dataset_filename}.json"), 'w') as json_file:
                json.dump(json_config, json_file, indent=4)

        if including_plots and self.__dataset is not None:
            for key in tqdm(self.__dataset_columns_name,
                            desc=str(f"Creating _{self.__dataset_name}_ columns plots"),
                            colour="blue"):
                if key in self.__dataset_analytics:
                    try:
                        os.mkdir(os.path.join(folder, "plots"))
                    except:
                        pass
                    self.__save_plots(path=os.path.join(folder, "plots"),
                                      column=self.__dataset_analytics[key])
        pd.DataFrame(self.__dataset).to_csv(os.path.join(folder, f"{dataset_filename}.csv"),
                                            index=False,
                                            sep=self.__delimiter,
                                            encoding=self.__encoding)

    def to_csv(self,
               file_name: str = None,
               path_to_saving_folder: str = None,
               delimeter: str = None,
               encoding: str = None) -> None:
        """
        This method saves pd.DataFrame to .csv file
        :param file_name: File name
        :param path_to_saving_folder: The path to the folder where the file will be saved
        :param delimeter: Symbol-split in a .csv/.tsv file
        :param encoding: Explicit indication of the .csv/.tsv file encoding
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if file_name is not None and not file_name.endswith((".csv", ".tsv")):
            raise Exception("The dataset format should be '.csv' or '.tsv'!")
        if file_name is None:
            file_name = f"{self.__dataset_name}.csv"
        if path_to_saving_folder is None:
            path_to_saving_folder = self.__dataset_save_path
        if encoding is not None and isinstance(encoding, str):
            self.set_encoding(encoding=encoding)
        if delimeter is not None and isinstance(delimeter, str):
            self.set_delimiter(delimiter=delimeter)
        pd.DataFrame(self.__dataset).to_csv(os.path.join(path_to_saving_folder, file_name),
                                            index=False,
                                            sep=self.__delimiter,
                                            encoding=self.__encoding)

    def to_excel(self,
                 file_name: str = None,
                 path_to_saving_folder: str = None,
                 sheet_name: str = None) -> None:
        """
        This method saves pd.DataFrame to excel file
        :param file_name: File name
        :param path_to_saving_folder: The path to the folder where the file will be saved
        :param sheet_name: Name of sheet in excel file
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if not file_name.endswith(".xlsx"):
            raise Exception("The dataset format should be '.xlsx'!")
        if file_name is None:
            file_name = f"{self.__dataset_name}.xlsx"
        if path_to_saving_folder is None:
            path_to_saving_folder = self.__dataset_save_path
        if sheet_name is not None:
            sheet_name = sheet_name
        elif self.__dataset_file is None:
            sheet_name = self.__dataset_name
        else:
            sheet_name = os.path.basename(self.__dataset_file).replace(".csv", "").replace(".xlsx", "")
        pd.DataFrame(self.__dataset).to_excel(os.path.join(path_to_saving_folder, file_name),
                                              index=False,
                                              sheet_name=sheet_name)



    # def __save_plots(self, path: str, column: DataSetNumColumn):
    #     if column.get_is_num_stat():
    #         self.__save_min_average_max_plot(path=path,
    #                                          column=column)
    #         if column.num_stat.get_is_normal_distribution():
    #             self.__save_normal_distribution_plot(path=path,
    #                                                  column=column)
    #
    # def __save_normal_distribution_plot(self, path: str, column: DataSetNumColumn):
    #     plot_title = f'Normal Distribution of _{column.get_column_name()}_'
    #     math_exp_v = column.num_stat.normal_distribution.get_math_expectation()
    #     math_sigma_v = column.num_stat.normal_distribution.get_math_sigma()
    #
    #     values_plot = column.get_values()
    #     math_expectation = len(values_plot) * [column.num_stat.normal_distribution.get_math_expectation()]
    #     plt.title(plot_title)
    #     plt.plot(values_plot, 'b-', label=f"Values count={len(values_plot)}")
    #     plt.plot(math_expectation, 'b--', label=f"Expectation(Sigma)={math_exp_v}({math_sigma_v})")
    #
    #     plt.plot(len(values_plot) * [math_exp_v + 3 * math_sigma_v], 'r--',
    #              label=f"Moda + 3 * sigma={math_exp_v + 3 * math_sigma_v}")
    #
    #     plt.plot(len(values_plot) * [math_exp_v + 2 * math_sigma_v], 'y--',
    #              label=f"Moda + 2 * sigma={math_exp_v + 2 * math_sigma_v}")
    #
    #     plt.plot(len(values_plot) * [math_exp_v + 1 * math_sigma_v], 'g--',
    #              label=f"Moda + 1 * sigma={math_exp_v + 1 * math_sigma_v}")
    #
    #     plt.plot(len(values_plot) * [math_exp_v - 1 * math_sigma_v], 'g--',
    #              label=f"Moda - 1 * sigma={math_exp_v - 1 * math_sigma_v}")
    #
    #     plt.plot(len(values_plot) * [math_exp_v - 2 * math_sigma_v], 'y--',
    #              label=f"Moda - 2 * sigma={math_exp_v - 2 * math_sigma_v}")
    #
    #     plt.plot(len(values_plot) * [math_exp_v - 3 * math_sigma_v], 'y--',
    #              label=f"Moda - 2 * sigma={math_exp_v - 3 * math_sigma_v}")
    #
    #     plt.legend(loc='best')
    #     if path is not None:
    #         if not os.path.exists(path):  # Надо что то с путём что то адекватное придумать
    #             raise Exception("The specified path was not found!")
    #         plt.savefig(os.path.join(path, plot_title + ".png"))
    #     plt.close()
    #
    # def __save_min_average_max_plot(self, path: str, column: DataSetNumColumn):
    #     plot_title = f'Numerical Indicators of _{column.get_column_name()}_'
    #     values_plot = column.get_values()
    #     max_plot = len(values_plot) * [column.get_max()]
    #     min_plot = len(values_plot) * [column.get_min()]
    #     average_plot = len(values_plot) * [column.get_mean()]
    #     plt.title(plot_title)
    #     plt.plot(values_plot, 'b-', label=f"Values count={len(values_plot)}")
    #     plt.plot(max_plot, 'r-', label=f"Max value={column.get_max()}")
    #     plt.plot(min_plot, 'r--', label=f"Min value={column.get_min()}")
    #     plt.plot(average_plot, 'g--', label=f"Average value={column.get_mean()}")
    #     plt.legend(loc='best')
    #     if path is not None:
    #         if not os.path.exists(path):  # Надо что то с путём что то адекватное придумать
    #             raise Exception("The specified path was not found!")
    #         plt.savefig(os.path.join(path, plot_title + ".png"))
    #     plt.close()

    def __get_column_type(self, column_name: str) -> ColumnType:
        """
        This method learns the column type
        :param column_name: Name of DataSet column
        """
        types = []
        for i in range(len(self.__dataset)):
            column_type = type(self.__dataset.at[i, column_name]).__name__
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

    def __read_dataset_info_from_json(self, data) -> None:
        """
        This method reads config and statistics info from .json file
        :param data: json data
        """
        self.__dataset_file = data["dataset_filename"]
        self.__dataset_columns_name = data["columns_names"]
        self.__dataset_columns_name_count = data["columns_count"]
        self.__dataset_len = data["rows"]
        self.__delimiter = data["delimiter"]
        self.__encoding = data["encoding"]
        for dk in self.__dataset_columns_name:
            self.__dataset_analytics[dk] = DataSetColumn(column_name=dk)
            self.__dataset_analytics[dk].get_from_json(data=data["columns"][dk],
                                                       values=self.__dataset[dk].values)

    def __update_dataset_base_info(self) -> None:
        """
        This method updates the basic information about the dataset
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if self.__dataset is not None:
            self.__dataset_len = len(self.__dataset)
            self.__dataset_columns_name = self.__dataset.keys()
            self.__dataset_columns_name_count = len(self.__dataset.keys())

    @staticmethod
    def __dif_lists_index(list_a: list, list_b: list) -> List[int]:
        diff = []
        for i in range(min(len(list_a), len(list_b))):
            if list_a[i] != list_b[i]:
                diff.append(i)
        return diff

    @staticmethod
    def __read_from_csv(filename: str,
                        delimiter: str,
                        encoding: str = 'utf-8') -> pd.DataFrame:
        """
        This method reads the dataset from a .csv file
        :param filename: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        """
        return pd.read_csv(filename,
                           encoding=encoding,
                           delimiter=delimiter)

    @staticmethod
    def __read_from_xlsx(filename: str,
                         sheet_name: str) -> pd.DataFrame:
        """
        This method reads the dataset from a .csv file
        :param filename: The name of the .csv file
        """
        for engine in ["xlrd", "openpyxl", "odf", "pyxlsb"]:
            try:
                return pd.read_excel(filename, sheet_name=sheet_name, engine=engine)
            except:
                pass

    @staticmethod
    def get_excel_sheet_names(excel_file: str) -> List[str]:
        """
        This method loads the dataset into the DataSet class
        :param excel_file: The name of the excel file
        """
        if excel_file is not None:  # Checking that the uploaded file has the .csv format
            if not excel_file.endswith(tuple(DataSet.supported_formats)):
                raise Exception(f"The dataset format should be {', '.join(DataSet.supported_formats)}!")
        return pd.ExcelFile(excel_file).sheet_names


def merge_two_dicts(dict1: dict, dict2: dict) -> dict:
    """
    This method merge two dicts
    :param dict1: First dict to merge
    :param dict2: Second dict to merge
    :return: dict
    """
    return {**dict1, **dict2}
