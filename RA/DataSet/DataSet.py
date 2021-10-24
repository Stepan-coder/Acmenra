import os
import sys
import json
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from typing import Any
from prettytable import PrettyTable

from RA.DataSet.DataSetColumnNum import *
from RA.DataSet.DataSetColumnStr import *


class DataSet:
    def __init__(self, dataset_name: str, show: bool = False) -> None:
        """
        This is an init method
        :param dataset_name: User nickname of dataset
        :param show: Do I need to show what is happening
        :return None
        """
        self.__dataset_name = dataset_name
        self.__show = show
        self.__delimiter = ","
        self.__encoding = 'utf-8'
        self.__is_dataset_loaded = False
        self.__dataset = None  # Dataset in pd.DataFrame
        self.__dataset_len = None  # Number of records in the dataset
        self.__dataset_keys = None  # Column names in the dataset
        self.__dataset_keys_count = None  # Number of columns in the dataset
        self.__dataset_file = None
        self.__dataset_save_path = None
        self.__dataset_analytics = {}

    def __len__(self) -> int:
        if self.__dataset is not None:
            return len(self.__dataset)
        else:
            return 0

    def __str__(self):
        table = PrettyTable()
        is_dataset = True if self.__dataset is not None and self.__dataset_len > 0 else False
        table.title = f"{'Empty ' if not is_dataset else ''}DataSet \"{self.__dataset_name}\""
        table.field_names = ["Column name", "Type", "Data type", "Count", "Count unique", "NaN count"]
        if is_dataset:
            for key in self.__dataset_keys:
                column = self.get_column_statinfo(column_name=key, extended=False)

                if column.get_dtype(threshold=0.15) == 'variable':
                    dtype = "\033[32m {}\033[0m".format(column.get_dtype(threshold=0.15))
                else:
                    dtype = "\033[31m {}\033[0m".format(column.get_dtype(threshold=0.15))
                table.add_row([column.get_column_name(),
                               column.get_type(),
                               dtype,
                               column.get_count(),
                               column.get_unique_count(),
                               column.get_nan_count()])
        return str(table)

    def get_supported_formats(self) -> List[str]:
        """
        This method returns a list of supported files
        :return: List[str]
        """
        return [".xls", ".xlsx", ".xlsm", ".xlt", ".xltx", ".xlsb", '.ots', '.ods']

    # SET_GET INIT PARAMS
    def set_name(self, dataset_name: str) -> None:
        """
        This method sets the project_name of the DataSet
        :param dataset_name: Name of this
        :return None
        """
        if not isinstance(dataset_name, str):
            raise Exception("The name must be a string!")
        if not len(dataset_name) > 0:
            raise Exception("The name must be longer than 0 characters!")
        self.__dataset_name = dataset_name

    def get_name(self) -> str:
        """
        This method returns the dataset name of the current DataSet
        :return: str
        """
        return self.__dataset_name

    def set_show(self, show: bool) -> None:
        """
        This method sets the show switcher
        :param show: Name of this
        :return None
        """
        if not isinstance(show, bool):
            raise Exception('The "show" parameter must be boolean!')
        self.__show = show

    def get_show(self) -> bool:
        """
        This method returns the show switcher
        :return: bool
        """
        return self.__show

    def set_delimiter(self, delimiter: str) -> None:
        """
        This method sets the delimiter character
        :param delimiter: Symbol-split in a .csv file
        :return None
        """
        if not isinstance(delimiter, str):
            raise Exception("The delimiter must be a one-symbol string!")
        if len(delimiter) > 1 or len(delimiter) == 0:
            raise Exception("A separator with a length of 1 character is allowed!")
        self.__delimiter = delimiter

    def get_delimiter(self) -> str:
        """
        This method returns a delimiter character
        :return Symbol-split in a .csv file
        """
        return self.__delimiter

    def set_encoding(self, encoding: str) -> None:
        """
        This method sets the encoding for the future export of the dataset
        :param encoding: Encoding for the dataset. Example: 'utf-8', 'windows1232'
        :return None
        """
        if not isinstance(encoding, str):
            raise Exception("The encoding must be a string!")
        if len(encoding) == 0:
            raise Exception("The name must be longer than 0 characters!")
        self.__encoding = encoding

    def get_encoding(self) -> str:
        """
        This method returns the encoding of the current dataset file
        :return: Encoding for the dataset. Example: 'utf-8', 'windows1232'
        """
        return self.__encoding

    def get_keys(self) -> List[str]:
        """
        This method return column names of dataset pd.DataFrame
        :return: List[str]
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        self.__update_dataset_base_info()
        return list(self.__dataset_keys)

    def set_keys_order(self, new_order_columns: List[str]) -> None:
        """
        This method set columns order
        :param new_order_columns: List of new order of columns
        :return None
        """
        if len(set(new_order_columns)) != len(new_order_columns):
            raise Exception(f"Column names should not be repeated!")
        for column in new_order_columns:
            if column not in self.__dataset_keys:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        for column in self.__dataset_keys:
            if column not in new_order_columns:
                raise Exception(f"The \"{column}\" column is missing!")
        self.__dataset = pd.DataFrame(self.__dataset, columns=new_order_columns)
        self.__update_dataset_base_info()

    def get_keys_count(self) -> int:
        """
        This method return count of column names of dataset pd.DataFrame
        :return: int
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet")
        self.__update_dataset_base_info()
        return self.__dataset_keys_count

    def set_to_field(self, column: str, index: int, value: Any) -> None:
        """
        This method gets the value from the dataset cell
        :param column: The name of the dataset column
        :param index: Index of the dataset string
        :param value: The value that we want to write
        :return None
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        if column not in self.__dataset_keys:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        if column in self.__dataset_analytics:
            self.__dataset_analytics.pop(column)
        self.__dataset.loc[index, column] = value

    def get_from_field(self, column: str, index: int) -> Any:
        """
        This method gets the value from the dataset cell
        :param column: The name of the dataset column
        :param index: Index of the dataset string
        :return: Any
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        if column not in self.__dataset_keys:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        return self.__dataset.at[index, column]

    def add_row(self, new_row: Dict[str, Any]) -> None:
        """
        This method adds a new row to the dataset
        :param new_row: The string to be added to the dataset. In dictionary format,
        where the key is the column name and the value is a list of values
        :return: None
        """
        if not self.__is_dataset_loaded:
            warnings.warn(f'The dataset was not loaded. '
                          f'An empty dataset was created with the columns {list(new_row.keys())}!', UserWarning)
            self.create_empty_dataset(columns_names=list(new_row.keys()))
        if len(set(new_row.keys())) != len(new_row.keys()):
            raise Exception(f"Column names should not be repeated!")
        for column in new_row:
            if column not in self.__dataset_keys:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        for column in self.__dataset_keys:
            if column not in new_row.keys():
                raise Exception(f"The \"{column}\" column is missing!")
        self.__dataset.loc[len(self.__dataset)] = [new_row[d] for d in self.__dataset_keys]
        for key in self.__dataset_keys:
            if key in self.__dataset_analytics:
                self.__dataset_analytics.pop(key)
        self.__update_dataset_base_info()

    def get_row(self, index: int) -> Dict[str, Any]:
        """
        This method returns a row of the dataset in dictionary format, where the keys are the column names and the
        values are the values in the columns
        :param index: Index of the dataset string
        :return: Dict[str, Any]
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if index < 0:
            raise Exception("The row index must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row index must be less than the number of rows in the dataset!")
        self.__update_dataset_base_info()
        result = {}
        for column in self.__dataset_keys:
            if column not in result:
                result[column] = self.get_from_field(column=column,
                                                     index=index)
        return result

    def delete_row(self, index: int) -> None:
        """
        This method delete row from dataset
        :param index: Index of the dataset string
        :return None
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if index < 0:
            raise Exception("The row index must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        self.__dataset = self.__dataset.drop(index=index)
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)
        for key in self.__dataset_keys:
            if key in self.__dataset_analytics:
                self.__dataset_analytics.pop(key)
        self.__update_dataset_base_info()

    def add_column(self, column_name: str, values: list, dif_len: bool = False) -> None:
        """
        This method adds the column to the dataset on the right
        :param column_name: String name of the column to be added
        :param values: List of column values
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        :return: None
        """
        if not self.__is_dataset_loaded:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if column_name in self.__dataset_keys:
            raise Exception(f"The '{column_name}' column already exists in the presented dataset!")
        if len(self.__dataset) != len(values) and len(self.__dataset) != 0:
            if not dif_len:
                raise Exception("The column and dataset must have the same size!")
        self.__dataset[column_name] = values
        self.__update_dataset_base_info()

    def get_column_values(self, column_name: str) -> list:
        """
        This method summarizes the values from the columns of the dataset and returns them as a list of tuples
        :param column_name: List of column names
        :return: List
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        return self.__dataset[column_name].tolist()

    def rename_column(self, column_name: str, new_column_name: str) -> None:
        """
        This method renames the column in the dataset
        :param column_name: The name of the column that we are renaming
        :param new_column_name: New name for the "column" column
        :return: None
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        if new_column_name in self.__dataset_keys:
            raise Exception(f"The \"{new_column_name}\" column does already exist in this dataset!")
        self.__dataset = self.__dataset.rename(columns={column_name: new_column_name})
        if column_name in self.__dataset_analytics:
            column_analytic = self.__dataset_analytics[column_name]
            self.__dataset_analytics.pop(column_name)
            self.__dataset_analytics[new_column_name] = column_analytic
            self.__dataset_analytics[new_column_name].set_column_name(new_column_name=new_column_name)
        self.__update_dataset_base_info()

    def delete_column(self, column_name: str) -> None:
        """
        This method removes the column from the dataset
        :param column_name: List of column names
        :return: None
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        self.__dataset = self.__dataset.drop([column_name], axis=1)
        if column_name in self.__dataset_analytics:
            self.__dataset_analytics.pop(column_name)
        self.__update_dataset_base_info()

    def set_column_types(self, new_column_types: type, exception: Dict[str, type] = None) -> None:
        """
        This method converts column types
        :param new_column_types: New type of dataset columns (excluding exceptions)
        :param exception: Fields that have a different type from the main dataset type
        :return None
        """
        for field in self.__dataset:
            if exception is None:
                self.set_column_type(column_name=field, new_column_type=new_column_types)
            else:
                if field in exception:
                    self.set_column_type(column_name=field, new_column_type=exception[field])
                else:
                    self.set_column_type(column_name=field, new_column_type=new_column_types)
        self.__update_dataset_base_info()

    def set_column_type(self, column_name: str, new_column_type: type) -> None:
        """
        This method converts column type
        :param column_name: The name of the column in which we want to change the type
        :param new_column_type: Field type
        :return None
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
            if self.__show:
                print(f"Convert DataSet field \'{column_name}\': {primary_type} -> {secondary_type}")
        else:
            raise Exception("There is no such column in the presented dataset!")

    def get_column_statinfo(self, column_name: str, extended: bool) -> DataSetColumnNum or DataSetColumnStr:
        """
        This method returns statistical analytics for a given column
        :param column_name: The name of the dataset column for which we output statistics
        :param extended: Responsible for calculating additional parameters
        :return: DataSetColumn
        """
        if not self.__is_dataset_loaded:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        if column_name not in self.__dataset_analytics:
            if self.__get_column_type(column_name=column_name).startswith("int") or \
                    self.__get_column_type(column_name=column_name).startswith("float"):
                self.__dataset_analytics[column_name] = DataSetColumnNum(column_name=column_name,
                                                                         values=list(self.__dataset[column_name]),
                                                                         extended=extended)
            elif self.__get_column_type(column_name=column_name).startswith("str"):
                self.__dataset_analytics[column_name] = DataSetColumnStr(column_name=column_name,
                                                                         values=list(self.__dataset[column_name]),
                                                                         extended=extended)
        return self.__dataset_analytics[column_name]

    def get_columns_stat_info(self) -> Dict[str, DataSetColumnNum]:
        """
        This method returns DataSet columns stat info
        :return: Dict["column_name", <DataSetColumn> class]
        """
        return self.__dataset_analytics

    def set_saving_path(self, path: str) -> None:
        """
        This method removes the column from the dataset
        :param path: The path to save the "DataSet" project
        :return: None
        """
        self.__dataset_save_path = path
    # /SET_GET INIT PARAMS

    def get_is_loaded(self) -> bool:
        """
        This method returns the state of this DataSet
        :return: bool
        """
        return self.__is_dataset_loaded

    def head(self, n: int = 5) -> None:
        """
        This method prints the first n rows
        :param n: Count of lines
        :return: None
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            n = len(self.__dataset)
        table = PrettyTable()
        table.title = self.__dataset_name
        table.field_names = self.__dataset_keys
        for i in range(n):
            this_row = self.get_row(index=i)
            table_row = []
            for column_name in self.__dataset_keys:
                column_type = self.get_column_statinfo(column_name=column_name, extended=False).get_type()
                if column_type.startswith("str") and isinstance(this_row[column_name], str) and \
                        len(this_row[column_name]) > 50:
                    table_row.append(f"{str(this_row[column_name])[:47]}...")
                else:
                    table_row.append(this_row[column_name])
            table.add_row(table_row)
        print(table)

    def tail(self, n: int = 5) -> None:
        """
        This method prints the last n rows
        :param n: Count of lines
        :return: None
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            n = len(self.__dataset)
        table = PrettyTable()
        table.title = self.__dataset_name
        table.field_names = self.__dataset_keys
        for i in range(self.__dataset_len - n, self.__dataset_len):
            this_row = self.get_row(index=i)
            table_row = []
            for column_name in self.__dataset_keys:
                column_type = self.get_column_statinfo(column_name=column_name, extended=False).get_type()
                if column_type.startswith("str") and isinstance(this_row[column_name], str) and \
                        len(this_row[column_name]) > 50:
                    table_row.append(f"{str(this_row[column_name])[:47]}...")
                else:
                    table_row.append(this_row[column_name])
            table.add_row(table_row)
        print(table)

    def fillna(self) -> None:
        """
        This method automatically fills in "null" values:
         For "int" -> 0.
         For "float" -> 0.0.
         For "str" -> "-".
        :return: None
        """
        for key in self.__dataset_keys:
            column_type = self.get_column_statinfo(column_name=key,
                                                   extended=False).get_type()
            if column_type.startswith('str'):
                self.__dataset[key] = self.__dataset[key].fillna(value="-")
            elif column_type.startswith('int') or column_type.startswith('float'):
                self.__dataset[key] = self.__dataset[key].fillna(value=0)
        self.update_dataset_info()

    def split(self, count: int) -> List:
        """
        This method automatically divides the DataSet into a list of DataSets with a maximum of "count" rows in each
        :param count: Maximal count of rows in new DataSets
        :return: List[DataSet]
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
            this_dataset.set_delimiter(delimiter=self.get_delimiter())
            this_dataset.set_encoding(encoding=self.get_encoding())
            result.append(this_dataset)
            counter += 1
        return result

    def sort_by_column(self, column_name: str, reverse: bool = False) -> None:
        """
        This method sorts the dataset by column "column_name"
        :param column_name: The name of the column by which the sorting will take place
        :param reverse: The parameter responsible for the sorting order (ascending/descending)
        :return: None
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        column_type = self.get_column_statinfo(column_name=column_name, extended=False).get_type()
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

    def get_DataFrame(self) -> pd.DataFrame:
        """
        This method return dataset as pd.DataFrame
        :return: pd.DataFrame
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been uploaded yet!")
        return self.__dataset

    def join_DataFrame(self, dataframe: pd.DataFrame, dif_len: bool = False) -> None:
        """
        This method attaches a new dataset to the current one (at right)
        :param dataframe: The pd.DataFrame to be attached to the current one
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        :return None
        """
        if self.__dataset is None:
            warnings.warn(f'The dataset was not loaded. An empty dataset was created!', UserWarning)
            self.create_empty_dataset()
        if len(dataframe) == 0:
            raise Exception("You are trying to add an empty dataset")
        if len(self.__dataset) != len(dataframe):
            if not dif_len:
                raise Exception("The pd.DataFrames must have the same size!")
        columns_names = list(self.__dataset.keys()) + list(dataframe.keys())
        if len(set(columns_names)) != len(columns_names):
            raise Exception("The current dataset and the new dataset have the same column names!")
        self.__dataset = self.__dataset.join(dataframe,
                                             how='outer')
        self.__update_dataset_base_info()

    def join_DataSet(self, dataset, dif_len: bool = False) -> None:
        """
        This method attaches a new dataset to the current one(at right)
        :param dataset: The DataSet object to be attached to the current one
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        :return None
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
        :return None
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
        :return None
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
        :return: None
        """
        self.__update_dataset_base_info()
        for key in self.__dataset_keys:
            is_extended = False
            if key in self.__dataset_analytics:
                is_extended = self.__dataset_analytics[key].get_is_extended()
            if self.__get_column_type(column_name=key).startswith("int") or \
                    self.__get_column_type(column_name=key).startswith("float"):
                self.__dataset_analytics[key] = DataSetColumnNum(column_name=key,
                                                                 values=self.__dataset[key],
                                                                 extended=is_extended)
            elif self.__get_column_type(column_name=key).startswith("str"):
                self.__dataset_analytics[key] = DataSetColumnStr(column_name=key,
                                                                 values=self.__dataset[key],
                                                                 extended=is_extended)

    # CREATE-LOAD-EXPORT DATASET
    def create_empty_dataset(self,
                             columns_names: list = None,
                             delimiter: str = ",",
                             encoding: str = 'utf-8'):
        """
        This method creates an empty dataset
        :param columns_names: List of column names
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return:
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if columns_names is not None:
            if len(set(columns_names)) != len(columns_names):
                raise Exception(f"Column names should not be repeated!")
            self.__dataset = pd.DataFrame(columns=columns_names)
        else:
            self.__dataset = pd.DataFrame()
        self.__delimiter = delimiter
        self.__encoding = encoding
        self.__is_dataset_loaded = True
        self.__update_dataset_base_info()

    def create_dataset_from_list(self,
                                 data: List[List[Any]],
                                 columns: List[str],
                                 delimiter: str = ",",
                                 encoding: str = 'utf-8') -> None:
        """
        This method creates a dataset from list
        :param data: List[List[Any] - lines of dataset]
        :param columns: List of column names
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return: None
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if len(data) == 0:
            raise Exception('The number of elements in "data" must be greater than 0!')
        if len(columns) == 0 or len(columns) != len(data[0]):
            raise Exception('There should be as many column names as there are columns in "data"!')
        self.__dataset = pd.DataFrame(data, columns=columns)
        self.set_delimiter(delimiter=delimiter)
        self.set_encoding(encoding=encoding)
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_DataFrame(self, dataframe: pd.DataFrame) -> None:
        """
        This method loads the dataset into the DataSet class
        :param dataframe: Explicitly specifying pd. DataFrame as a dataset
        :return None
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
        :return: None
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

    def get_excel_sheet_names(self, excel_file: str) -> List[str]:
        """
        This method loads the dataset into the DataSet class
        :param excel_file: The name of the excel file
        :return: List[str]
        """
        if excel_file is not None:  # Checking that the uploaded file has the .csv format
            if not excel_file.endswith(tuple(self.get_supported_formats())):
                raise Exception(f"The dataset format should be {', '.join(self.get_supported_formats())}!")
        return pd.ExcelFile(excel_file).sheet_names

    def load_excel_dataset(self,
                           excel_file: str,
                           sheet_name: str) -> None:
        """
        This method loads the dataset into the DataSet class
        :param sheet_name: Name of sheet in excel file
        :param excel_file: The name of the excel file
        :return: None
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if excel_file is not None:  # Checking that the uploaded file has the .csv format
            if not excel_file.endswith(tuple(self.get_supported_formats())):
                raise Exception(f"The dataset format should be {', '.join(self.get_supported_formats())}!")
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
        :return None
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
        :return: None
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if self.__show:
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
                           "columns_names": list(self.__dataset_keys),
                           "columns_count": self.__dataset_keys_count,
                           "rows": self.__dataset_len,
                           "delimiter": self.__delimiter,
                           "encoding": self.__encoding,
                           "columns": {}}
            for key in self.__dataset_keys:
                if key not in self.__dataset_analytics:
                    self.__dataset_analytics[key] = self.get_column_statinfo(column_name=key, extended=True)
                json_config["columns"] = merge_two_dicts(json_config["columns"],
                                                         self.__dataset_analytics[key].to_json())
            with open(os.path.join(folder, f"{dataset_filename}.json"), 'w') as json_file:
                json.dump(json_config, json_file, indent=4)

        if including_plots and self.__dataset is not None:
            for key in tqdm(self.__dataset_keys,
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
        :return: None
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
        :return: None
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

    # CREATE-LOAD-EXPORT DATASET

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

    def __get_column_type(self, column_name: str) -> str:
        """
        This method learns the column type
        :param column_name: Name of DataSet column
        :return: str
        """
        types = []
        for i in range(len(self.__dataset)):
            types.append(type(self.__dataset.at[i, column_name]).__name__)
        types = list(set(types))
        if len(types) == 1:
            return types[0]
        else:
            if len(types) == 2 and 'str' in types:
                return 'str'
            return "object"

    def __read_dataset_info_from_json(self, data) -> None:
        """
        This method reads config and statistics info from .json file
        :param data: json data
        :return None
        """
        self.__dataset_file = data["dataset_filename"]
        self.__dataset_keys = data["columns_names"]
        self.__dataset_keys_count = data["columns_count"]
        self.__dataset_len = data["rows"]
        self.__delimiter = data["delimiter"]
        self.__encoding = data["encoding"]
        for dk in self.__dataset_keys:
            self.__dataset_analytics[dk] = DataSetColumn(column_name=dk)
            self.__dataset_analytics[dk].get_from_json(data=data["columns"][dk],
                                                       values=self.__dataset[dk].values)

    def __update_dataset_base_info(self) -> None:
        """
        This method updates the basic information about the dataset
        :return None
        """
        if self.__dataset is None:
            print("dataset is None")
        if self.__dataset is not None:
            self.__dataset_len = len(self.__dataset)
            self.__dataset_keys = self.__dataset.keys()
            self.__dataset_keys_count = len(self.__dataset.keys())

    @staticmethod
    def __read_from_csv(filename: str,
                        delimiter: str,
                        encoding: str = 'utf-8') -> pd.DataFrame:
        """
        This method reads the dataset from a .csv file
        :param filename: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return: pd.DataFrame
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
        :return: pd.DataFrame
        """
        try:
            return pd.read_excel(filename, sheet_name=sheet_name, engine="xlrd")
        except:
            pass
        try:
            return pd.read_excel(filename, sheet_name=sheet_name, engine="openpyxl")
        except:
            pass
        try:
            return pd.read_excel(filename, sheet_name=sheet_name, engine="odf")
        except:
            pass
        try:
            return pd.read_excel(filename, sheet_name=sheet_name, engine="pyxlsb")
        except:
            pass


def merge_two_dicts(dict1: dict, dict2: dict) -> dict:
    """
    This method merge two dicts
    :param dict1: First dict to merge
    :param dict2: Second dict to merge
    :return: dict
    """
    return {**dict1, **dict2}
