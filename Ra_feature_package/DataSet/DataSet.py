import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict
from Ra_feature_package.DataSet.DataSetColumn import *

"""
TODO 
.head() - вывод красивой табличкой, сделать проверки на адекватные значения
.tail() - вывод красивой табличкой, сделать проверки на адекватные значения

"""

class DataSet:
    def __init__(self, dataset_project_name: str, show: bool = False):
        """
        This is an init method
        :param dataset_project_name: User nickname of dataset
        :param show: Do I need to show what is happening
        """
        self.__dataset_project_name = dataset_project_name
        self.__show = show
        self.__delimiter = ","
        self.__encoding = 'utf-8'
        self.__is_dataset_loaded = False
        self.__dataset = None  # Dataset in pd.DataFrame
        self.__dataset_len = None  # Number of records in the dataset
        self.__dataset_keys = None  # Column names in the dataset
        self.__dataset_keys_count = None  # Number of columns in the dataset
        self.__dataset_file = None
        self.__dataset_analytics = {}

    def __len__(self):
        return len(self.__dataset)

    def head(self, n: int = 5):
        """
        This method ...
        :param n: Count of lines
        :return:
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            raise Exception("Count of rows 'n' should be less, then length of dataset!")
        return self.__dataset.iloc[:n]

    def tail(self, n: int = 5):
        """
        This method ...
        :param n: Count of lines
        :return:
        """
        if self.__dataset is None:
            raise Exception("The dataset has not been loaded yet!")
        if n <= 0:
            raise Exception("Count of rows 'n' should be large, then 0!")
        if n > len(self.__dataset):
            raise Exception("Count of rows 'n' should be less, then length of dataset!")
        return self.__dataset.iloc[-n:]

    # set-get init params
    def set_project_name(self, project_name: str) -> None:
        """
        This method sets the project_name of the DataSet
        :param project_name: Name of this
        """
        self.__dataset_project_name = project_name

    def get_project_name(self) -> str:
        """
        This method returns the project_name of the current DataSet
        :return:
        """
        return self.__dataset_project_name

    def set_show(self, show: bool) -> None:
        """
        This method sets the show switcher
        :param show: Name of this
        """
        self.__show = show

    def get_show(self) -> bool:
        """
        This method returns the show switcher
        :return:
        """
        return self.__show

    def set_delimiter(self, delimiter: str) -> None:
        """
        This method sets the delimiter character
        :param delimiter: Symbol-split in a .csv file
        """
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
        """
        self.__encoding = encoding

    def get_encoding(self) -> str:
        """
        This method returns the encoding of the current dataset file
        :return: Encoding for the dataset. Example: 'utf-8', 'windows1232'
        """
        return self.__encoding

    #
    def get_keys(self) -> list:
        """
        This method return column names of dataset pd.DataFrame
        :return: Column names of dataset pd.DataFrame
        """
        if self.__dataset is not None:
            self.__update_dataset_base_info()
            return list(self.__dataset_keys)
        else:
            raise Exception("The dataset has not been loaded yet")

    def set_keys_order(self, new_order_columns: List[str]) -> None:
        """
        This method set columns order
        :param new_order_columns: List of new order of columns
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
        :return: Count of column names of dataset pd.DataFrame
        """
        if self.__dataset is not None:
            self.__update_dataset_base_info()
            return self.__dataset_keys_count
        else:
            raise Exception("The dataset has not been loaded yet")

    def get_from_field(self, column: str, index: int):
        """
        This method gets the value from the dataset cell
        :param column: The name of the dataset column
        :param index: Index of the dataset string
        :return: value
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        if column not in self.__dataset_keys:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        return self.__dataset.at[index, column]

    def set_to_field(self, column: str, index: int, value):
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
        if column not in self.__dataset_keys:
            raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        self.__dataset.loc[index, column] = value

    def add_row(self, new_row: dict):
        if len(set(new_row.keys())) != len(new_row.keys()):
            raise Exception(f"Column names should not be repeated!")
        for column in new_row:
            if column not in self.__dataset_keys:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        for column in self.__dataset_keys:
            if column not in new_row.keys():
                raise Exception(f"The \"{column}\" column is missing!")
        self.__dataset.loc[len(self.__dataset)] = [new_row[d] for d in self.__dataset_keys]
        self.__update_dataset_base_info()

    def get_row(self, index: int) -> dict:
        """
        This method returns a row of the dataset in dictionary format, where the keys are the column names and the
        values are the values in the columns
        :param index: Index of the dataset string
        :return:
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        self.__update_dataset_base_info()
        result = {}
        for column in self.__dataset_keys:
            if column not in result:
                result[column] = self.get_from_field(column=column,
                                                     index=index)
        return result

    def delete_row(self, index) -> None:
        """
        This method delete row from dataset
        :param index: Index of the dataset string
        """
        if index < 0:
            raise Exception("The string value must be greater than 0!")
        if index > self.__dataset_len:
            raise Exception("The row value must be less than the number of rows in the dataset!")
        self.__dataset = self.__dataset.drop(index=index)
        self.__dataset = self.__dataset.reset_index(level=0, drop=True)
        self.__update_dataset_base_info()

    def set_column(self, column_name: str, values: list):
        pass

    def get_column(self, columns: List[str]) -> pd.DataFrame:
        """
        This method summarizes the values from the columns of the dataset and returns them as a list of tuples
        :param columns: List of column names
        :return: Returns the selected columns
        """
        for column in columns:
            if column not in self.__dataset_keys:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        for column in columns:
            if column not in self.__dataset_keys:
                raise Exception(f"The \"{column}\" column does not exist in this dataset!")
        return pd.DataFrame(self.__dataset[columns])

    def get_column_info(self, column_name: str, extended: bool = False) -> DataSetColumn:
        """
        This method returns statistical analytics for a given column
        :param column_name: The name of the dataset column for which we output statistics
        :param normal_distribution: Responsible for calculating additional parameters
        :return:
        """
        if column_name not in self.__dataset_keys:
            raise Exception(f"The \"{column_name}\" column does not exist in this dataset!")
        if column_name not in self.__dataset_analytics:
            self.__dataset_analytics[column_name] = DataSetColumn(column_name=column_name,
                                                                  values=self.__dataset[column_name],
                                                                  normal_distribution=extended)
        return self.__dataset_analytics[column_name]

    def get_dataframe(self) -> pd.DataFrame:
        """
        This method return dataset as pd.DataFrame
        :return: dataset as pd.DataFrame
        """
        if self.__dataset is not None:
            return self.__dataset
        else:
            raise Exception("The dataset has not been uploaded yet!")

    def join_dataset(self, dataset: pd.DataFrame, dif_len: bool = False):
        """
        This method attaches a new dataset to the current one
        :param dataset: The dataset to be attached to the current one
        :param dif_len: The switch is responsible for maintaining the dimensionality of datasets
        """
        if len(dataset) == 0:
            raise Exception("You are trying to add an empty dataset")
        if len(self.__dataset) != len(dataset):
            if not dif_len:
                raise Exception("The pd.DataFrames must have the same size!")
        columns_names = list(self.__dataset.keys()) + list(dataset.keys())
        if len(set(columns_names)) != len(columns_names):
            raise Exception("The current dataset and the new dataset have the same column names!")
        self.__dataset = self.__dataset.join(dataset)
        self.__update_dataset_base_info()

    def set_field_types(self, new_fields_type: type = None, exception: Dict[str, type] = None):
        """
        This method converts column types
        :param new_fields_type: New type of dataset columns (excluding exceptions)
        :param exception: Fields that have a different type from the main dataset type
        """
        if new_fields_type is None and exception is None:
            raise Exception("One of the parameters \'new_fields_type\' or \'exception\' must not be empty!")
        for field in self.__dataset:
            if field not in exception and new_fields_type is not None:
                self.set_field_type(field_name=field,
                                    new_field_type=new_fields_type)
            elif field in exception:
                self.set_field_type(field_name=field,
                                    new_field_type=exception[field])

    def set_field_type(self, field_name: str, new_field_type: type):
        """
        This method converts column type
        :param field_name: The name of the column in which we want to change the type
        :param new_field_type: Field type
        """
        if new_field_type != str and new_field_type != int and new_field_type != float:
            raise Exception(f"'{new_field_type}' is an invalid data type for conversion. Valid types: int, float, str")
        if field_name in self.__dataset:
            primary_type = type(self.__dataset[field_name][0]).__name__
            if new_field_type == float or new_field_type == int:
                self.__dataset[field_name] = self.__dataset[field_name].replace(",", ".", regex=True) \
                    .replace(" ", "", regex=True) \
                    .replace(u'\xa0', u'', regex=True) \
                    .fillna(0)
                self.__dataset[field_name] = self.__dataset[field_name].astype(float)
                if new_field_type == int:
                    self.__dataset[field_name] = self.__dataset[field_name].astype(int)
            else:
                self.__dataset[field_name] = self.__dataset[field_name].astype(new_field_type)
            secondary_type = type(self.__dataset[field_name][0]).__name__
            if self.__show:
                print(f"Convert DataSet field \'{field_name}\': {primary_type} -> {secondary_type}")
        else:
            raise Exception("There is no such column in the presented dataset!")

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

    def load_DataFrame(self, dataframe: pd.DataFrame):
        """
        This method loads the dataset into the DataSet class
        :param dataset: Explicitly specifying pd. DataFrame as a dataset
        """
        if self.__is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        self.__dataset = dataframe
        self.__update_dataset_base_info()
        self.__is_dataset_loaded = True

    def load_csv_dataset(self,
                         csv_file: str,
                         delimiter: str,
                         encoding: str = 'utf-8'):
        """
        This method loads the dataset into the DataSet class
        :param csv_file: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return:
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

    def load_dataset_project(self,
                             dataset_project_folder: str,
                             json_config_filename: str):
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
               including_plots: bool = False):
        """
        This method exports the dataset as DataSet Project
        :param dataset_name: New dataset name (if the user wants to specify another one)
        :param dataset_folder: The folder to place the dataset files in
        :param including_json: Responsible for the export .the json config file together with the dataset
        :return:
        """
        if self.__show:
            # print(f"Saving DataSet \'{self.__dataset_project_name}\'...")
            pass
        if dataset_name is not None:
            dataset_filename = dataset_name
        elif self.__dataset_file is None:
            dataset_filename = self.__dataset_project_name
        else:
            dataset_filename = os.path.basename(self.__dataset_file).replace(".csv", "")

        if not os.path.exists(os.path.join(dataset_folder, dataset_filename)):
            os.makedirs(os.path.join(dataset_folder, dataset_filename))

        folder = ""
        if dataset_folder is not None:
            if os.path.exists(dataset_folder):
                folder = os.path.join(dataset_folder, dataset_filename)
            else:
                raise Exception("Ошибка!")
        else:
            folder = dataset_filename

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
                    self.update_dataset_analytics()
                json_config["columns"] = merge_two_dicts(json_config["columns"],
                                                         self.__dataset_analytics[key].to_json())
            with open(os.path.join(folder, f"{dataset_filename}.json"), 'w') as json_file:
                json.dump(json_config, json_file, indent=4)

        if including_plots and self.__dataset is not None:
            for key in tqdm(self.__dataset_keys,
                            desc=str(f"Creating _{self.__dataset_project_name}_ columns plots"),
                            colour="blue"):
                if key in self.__dataset_analytics:
                    try:
                        os.mkdir(os.path.join(folder, "plots"))
                    except:
                        pass
                    self.__save_plots(path=os.path.join(folder, "plots"),
                                      column=self.__dataset_analytics[key])

        self.__dataset.to_csv(os.path.join(folder, f"{dataset_filename}.csv"),
                              index=False,
                              sep=self.__delimiter,
                              encoding=self.__encoding)

    def __save_plots(self, path: str,  column: DataSetColumn):
        if column.get_is_num_stat():
            self.__save_min_average_max_plot(path=path,
                                             column=column)

            if column.num_stat.get_is_normal_distribution():
                self.__save_normal_distribution_plot(path=path,
                                                     column=column)

    def __save_normal_distribution_plot(self, path: str, column: DataSetColumn):
        plot_title = f'Normal Distribution of _{column.get_column_name()}_'
        math_exp_v = column.num_stat.normal_distribution.get_math_expectation()
        math_sigma_v = column.num_stat.normal_distribution.get_math_sigma()

        values_plot = column.get_values()
        math_expectation = len(values_plot) * [column.num_stat.normal_distribution.get_math_expectation()]
        plt.title(plot_title)
        plt.plot(values_plot, 'b-', label=f"Values count={len(values_plot)}")
        plt.plot(math_expectation, 'b--', label=f"Expectation(Sigma)={math_exp_v}({math_sigma_v})")

        plt.plot(len(values_plot) * [math_exp_v + 3 * math_sigma_v], 'r--',
                 label=f"Moda + 3 * sigma={math_exp_v + 3 * math_sigma_v}")

        plt.plot(len(values_plot) * [math_exp_v + 2 * math_sigma_v], 'y--',
                 label=f"Moda + 2 * sigma={math_exp_v + 2 * math_sigma_v}")

        plt.plot(len(values_plot) * [math_exp_v + 1 * math_sigma_v], 'g--',
                 label=f"Moda + 1 * sigma={math_exp_v + 1 * math_sigma_v}")

        plt.plot(len(values_plot) * [math_exp_v - 1 * math_sigma_v], 'g--',
                 label=f"Moda - 1 * sigma={math_exp_v - 1 * math_sigma_v}")

        plt.plot(len(values_plot) * [math_exp_v - 2 * math_sigma_v], 'y--',
                 label=f"Moda - 2 * sigma={math_exp_v - 2 * math_sigma_v}")

        plt.plot(len(values_plot) * [math_exp_v - 3 * math_sigma_v], 'y--',
                 label=f"Moda - 2 * sigma={math_exp_v - 3 * math_sigma_v}")

        plt.legend(loc='best')
        if path is not None:
            if not os.path.exists(path):  # Надо что то с путём что то адекватное придумать
                raise Exception("The specified path was not found!")
            plt.savefig(os.path.join(path, plot_title + ".png"))
        plt.close()

    def __save_min_average_max_plot(self, path: str, column: DataSetColumn):
        plot_title = f'Numerical Indicators of _{column.get_column_name()}_'
        values_plot = column.get_values()
        max_plot = len(values_plot) * [column.num_stat.get_max()]
        min_plot = len(values_plot) * [column.num_stat.get_min()]
        average_plot = len(values_plot) * [column.num_stat.get_average()]
        plt.title(plot_title)
        plt.plot(values_plot, 'b-', label=f"Values count={len(values_plot)}")
        plt.plot(max_plot, 'r-', label=f"Max value={column.num_stat.get_max()}")
        plt.plot(min_plot, 'r--', label=f"Min value={column.num_stat.get_min()}")
        plt.plot(average_plot, 'g--', label=f"Average value={column.num_stat.get_average()}")
        plt.legend(loc='best')
        if path is not None:
            if not os.path.exists(path):  # Надо что то с путём что то адекватное придумать
                raise Exception("The specified path was not found!")
            plt.savefig(os.path.join(path, plot_title + ".png"))
        plt.close()

    def __read_dataset_info_from_json(self, data):
        """
        This method reads config and statistics info from .json file
        :param data: json data
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
        self.update_dataset_analytics()

    def __update_dataset_base_info(self):
        """
        This method updates the basic information about the dataset
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
        :return: The dataframe read from the file
        """
        return pd.read_csv(filename,
                           encoding=encoding,
                           delimiter=delimiter)


def merge_two_dicts(dict1: dict, dict2: dict) -> dict:
    """
    This method merge two dicts
    :param dict1: First dict to merge
    :param dict2: Second dict to merge
    :return: result merged dict
    """
    return {**dict1, **dict2}