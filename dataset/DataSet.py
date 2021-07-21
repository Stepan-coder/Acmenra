import os
import json
import math
import numpy as np
import pandas as pd

from typing import Dict
from DataSetColumn import *


class DataSet:
    def __init__(self, dataset_project_name: str, show: bool = False):
        """
        This is an init methon
        :param dataset_project_name: User nickname of dataset
        :param show: Do I need to show what is happening
        """
        self.dataset_project_name = dataset_project_name
        self.show = show
        self.is_dataset_loaded = False
        self.dataset = None  # Dataset in pd.DataFrame
        self.dataset_len = None  # Number of records in the dataset
        self.dataset_keys = None  # Column names in the dataset
        self.dataset_keys_count = None  # Number of columns in the dataset
        self.dataset_file = None
        self.delimiter = ","
        self.dataset_analytics = {}

    def __len__(self):
        return len(self.dataset)

    def get_dataset_analytics(self, normal_distribution: bool = False):
        """
        This method calculates column metrics
        :return:
        """
        if self.dataset is not None:
            for key in self.dataset_keys:
                self.dataset_analytics[key] = DataSetFieldAnalytics(column_name=key,
                                                                    values=self.dataset[key],
                                                                    normal_distribution=normal_distribution)

    def set_field_types(self, new_fields_type: type = None, exception: Dict[str, type] = None):
        """
        This method converts column types
        :param new_fields_type: New type of dataset columns (excluding exceptions)
        :param exception: Fields that have a different type from the main dataset type
        """
        if new_fields_type is None and exception is None:
            raise Exception("One of the parameters \'new_fields_type\' or \'exception\' must not be empty!")
        for field in self.dataset:
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
        if field_name in self.dataset:
            primary_type = type(self.dataset[field_name][0]).__name__
            if new_field_type == float or new_field_type == int:
                self.dataset[field_name] = self.dataset[field_name].replace(",", ".", regex=True) \
                    .replace(" ", "", regex=True) \
                    .replace(u'\xa0', u'', regex=True) \
                    .fillna(0)
                self.dataset[field_name] = self.dataset[field_name].astype(float)
                if new_field_type == int:
                    self.dataset[field_name] = self.dataset[field_name].astype(int)
            else:
                self.dataset[field_name] = self.dataset[field_name].astype(new_field_type)
            secondary_type = type(self.dataset[field_name][0]).__name__
            if self.show:
                print(f"Convert DataSet field \'{field_name}\': {primary_type} -> {secondary_type}")
        else:
            raise Exception("There is no such column in the presented dataset!")

    def load_dataset(self,
                     dataset: pd.DataFrame):
        """
        This method loads the dataset into the DataSet class
        :param dataset: Explicitly specifying pd. DataFrame as a dataset
        """
        if self.is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        self.dataset = dataset
        self._update_dataset_base_info()
        self.is_dataset_loaded = True

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
        if self.is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if csv_file is not None:  # Checking that the uploaded file has the .csv format
            if not csv_file.endswith(".csv"):
                raise Exception("The dataset format should be '.csv'!")
        if csv_file is not None and delimiter is None:  # Checking that a separator is specified
            raise Exception("When loading a dataset from a .csv file, you must specify a separator character!")
        self.delimiter = delimiter
        self.dataset_file = csv_file
        self.dataset = self._read_from_csv(filename=csv_file,
                                           delimiter=delimiter,
                                           encoding=encoding)
        self._update_dataset_base_info()
        self.is_dataset_loaded = True

    def load_dataset_project(self,
                             dataset_project_folder: str,
                             json_config_filename: str):
        """
        This method loads the dataset into the DataSet class
        :param dataset_project_folder: The path to the .csv file
        :param json_config_filename:
        """
        if self.is_dataset_loaded:
            raise Exception("The dataset is already loaded!")
        if not json_config_filename.endswith(".json"):
            raise Exception("The file must be in .json format!")
        if not os.path.exists(dataset_project_folder):
            raise Exception("The specified path was not found!")

        self.dataset = None
        with open(f"{dataset_project_folder}\\{json_config_filename}", 'r') as jsonfile:
            dataset_info = json.load(jsonfile)
            jsonfile.close()
            self._read_dataset_info_from_json(dataset_info)
        self.is_dataset_loaded = True

    def export(self,
               dataset_name: str = None,
               dataset_folder: str = None,
               including_json: bool = True):
        """
        This method exports the dataset
        :param dataset_name: New dataset name (if the user wants to specify another one)
        :param dataset_folder: The folder to place the dataset files in
        :param including_json: Responsible for the export .the json config file together with the dataset
        :return:
        """
        if dataset_name is not None:
            dataset_filename = dataset_name
        elif self.dataset_file is None:
            dataset_filename = self.dataset_project_name
        else:
            dataset_filename = os.path.basename(self.dataset_file).replace(".csv", "")

        folder = ""
        if dataset_folder is not None:
            if os.path.exists(dataset_folder):
                folder = f"{dataset_folder}\\"
            else:
                raise Exception("The specified path was not found!")

        if not os.path.exists(f"{dataset_filename}"):
            os.makedirs(f"{dataset_filename}")

        if including_json:
            json_config = {"dataset_filename": f"{dataset_filename}.csv",
                           "columns_names": list(self.dataset_keys),
                           "columns_count": self.dataset_keys_count,
                           "rows": self.dataset_len,
                           "delimiter": self.delimiter,
                           "columns": {}}
            if self.dataset is not None:
                for key in self.dataset_keys:
                    json_config["columns"] = merge_two_dicts(json_config["columns"],
                                                             self.dataset_analytics[key].to_json())
                with open(f"{folder}{dataset_filename}\\{dataset_filename}.json", 'w') as jsonfile:
                    json.dump(json_config, jsonfile, indent=4)
        self.dataset.to_csv(f"{folder}{dataset_filename}\\{dataset_filename}.csv", index=False)

    def _read_dataset_info_from_json(self, data):
        self.dataset_file = data["dataset_filename"]
        self.dataset_keys = data["columns_names"]
        self.dataset_keys_count = data["columns_count"]
        self.dataset_len = data["rows"]
        self.delimiter = data["delimiter"]
        for dk in self.dataset_keys:
            self.dataset_analytics[dk] = DataSetFieldAnalytics(column_name=dk)
            self.dataset_analytics[dk].get_from_json(data=data["columns"][dk])

    def _update_dataset_base_info(self):
        """
        This method updates the basic information about the dataset
        """
        if self.dataset is not None:
            self.dataset_len = len(self.dataset)
            self.dataset_keys = self.dataset.keys()
            self.dataset_keys_count = len(self.dataset.keys())

    @staticmethod
    def _read_from_csv(filename: str,
                       delimiter: str,
                       encoding: str = 'utf-8'):
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


def merge_two_dicts(dict1, dict2):
    """
    This method merge two dicts
    :param dict1: First dict to merge
    :param dict2: Second dict to merge
    :return: result merged dict
    """
    return {**dict1, **dict2}
