import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict


class DataSet:
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 dataset_project_folder: str = None,
                 csv_filename: str = None,
                 delimiter: str = None,
                 show: bool = False):
        self.dataset = None  # Dataset in pd.DataFrame
        self.dataset_project_folder = dataset_project_folder
        self.csv_filename = csv_filename
        self.delimiter = delimiter
        self.show = show
        self.dataset_len = None  # Number of records in the dataset
        self.dataset_keys = None  # Column names in the dataset
        self.dataset_keys_count = None  # Number of columns in the dataset
        self.dataset_analytics = {}
        if dataset is not None or dataset_project_folder is not None or csv_filename is not None:
            self.load_dataset(dataset=dataset,
                              dataset_project_folder=dataset_project_folder,
                              csv_filename=csv_filename,
                              delimiter=delimiter)
        self.get_dataset_analytics(normal_distribution=False)

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
            raise Exception("One of the parameters \'new_fields_type\' or \'exception\' must not be empty")
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
            raise Exception("There is no such column in the presented dataset")

    def load_dataset(self,
                     dataset: pd.DataFrame = None,
                     dataset_project_folder: str = None,
                     dataset_project_name: str = None,
                     csv_filename: str = None,
                     delimiter: str = None,
                     encoding: str = 'utf-8'):
        """
        This method loads the dataset into the DataSet class
        :param dataset: Explicitly specifying pd. DataFrame as a dataset
        :param dataset_project_folder: The path to the .csv file
        :param csv_filename: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return:
        """
        if csv_filename is not None:
            if not csv_filename.endswith(".csv"):
                raise Exception("The dataset format should be '.csv'")
        if csv_filename is not None and delimiter is None:
            raise Exception("When loading a dataset from a .csv file, you must specify a separator character!")
        if (dataset is not None and dataset_project_name is not None) \
                or (dataset is not None and csv_filename is not None) \
                or (dataset_project_name is not None and dataset is not None) \
                or (dataset_project_name is not None and csv_filename is not None and dataset is not None):
            raise Exception("You need to select only one of the ways to load the dataset!")
        if dataset_project_name is None and csv_filename is None and dataset is None:
            raise Exception("Please choose one of the ways to download the dataset")

        if dataset is not None:  # If pd.DataFrame was uploaded to us, we believe that this is the dataset
            self.dataset = dataset
        elif csv_filename is not None and delimiter is not None:  # Otherwise, we load the dataset from csv
            self.dataset = self._read_from_csv(filename=csv_filename,
                                               delimiter=delimiter,
                                               encoding=encoding)
        elif dataset_project_folder is not None and dataset_project_name is not None:  # Otherwise, we believe that the dataset will be handed over to us later
            self.dataset = None
            folder = ""
            if dataset_project_folder is not None and dataset_project_folder != "":
                folder = dataset_project_folder + "\\"
            with open(f"{folder}{dataset_project_name}\\{dataset_project_name}.json", 'r') as jsonfile:
                dataset_info = json.load(jsonfile)
                jsonfile.close()
                print(dataset_info)
        self._update_dataset_base_info()

    def export(self,
               dataset_name: str,
               dataset_folder: str = None,
               including_json: bool = True):
        dataset_name = f"dataset_{dataset_name}"
        folder = ""
        if dataset_folder is not None:
            if os.path.exists(dataset_folder):
                folder = f"{dataset_folder}\\"
            else:
                raise Exception("The specified path was not found!")

        if not os.path.exists(f"{dataset_name}"):
            os.makedirs(f"{dataset_name}")

        if including_json:
            json_config = {"dataset_name": dataset_name[8:], "columns_names": list(self.dataset_keys),
                           "columns_count": self.dataset_keys_count, "rows": self.dataset_len, "columns": {}}
            if self.dataset is not None:
                for key in self.dataset_keys:
                    json_config["columns"] = merge_two_dicts(json_config["columns"],
                                                             self.dataset_analytics[key].to_json())
                with open(f"{folder}{dataset_name}\\{dataset_name}.json", 'w') as jsonfile:
                    json.dump(json_config, jsonfile, indent=4)
        self.dataset.to_csv(f"{folder}{dataset_name}\\{dataset_name}.csv", index=False)

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


class DataSetFieldAnalytics:
    def __init__(self,
                 column_name: str,
                 values: list,
                 normal_distribution: bool = False,
                 rounding_factor: int = 2):
        self.column_name = column_name
        self.values = values
        self.normal_distribution = normal_distribution
        self.rounding_factor = rounding_factor
        self.field_type = type(values[0]).__name__
        self.count = len(values)
        self.count_unique = len(list(set(values)))
        self.nan_count = self._get_nan_count()
        if self.field_type.startswith("int") or self.field_type == "float":
            self.min = min(values)
            self.max = max(values)
            self.average = sum(values) / self.count
            if self.normal_distribution:  # Exporting additional statistics
                math_rasp = self._get_math_rasp()
                self.math_moda = self._get_math_moda(math_rasp)
                self.math_wait = self._get_math_wait(math_rasp)
                self.math_dispersion = self._get_math_dispersion(math_rasp)
                self.math_sigma = math.sqrt(self.math_dispersion)

    def __str__(self):
        """
        Displaying information about the column
        """
        text = f"DataSetField: \'{self.column_name}\'\n"
        text += f"  -Name: \'{self.column_name}\'\n"
        text += f"  -Type: \'{self.field_type}\'\n"
        text += f"  -Count: {self.count}\n"
        text += f"  -Count unique: {self.count_unique}\n"
        text += f"  -Count NaN: {self.nan_count}\n"
        if self.field_type.startswith("int") or self.field_type == "float":
            text += f"  Numerical indicators:\n"
            text += f"      -Min: {self.min}\n"
            text += f"      -Max: {self.max}\n"
            text += f"      -Average: {round(self.average, self.rounding_factor)}\n"
            if self.normal_distribution:
                text += f"  Normal distribution:\n"
                text += f"      -Moda: {self.math_moda}\n"
                text += f"      -Wait: {round(self.math_wait, self.rounding_factor)}\n"
                text += f"      -Dispersion: {round(self.math_dispersion, self.rounding_factor)}\n"
                text += f"      -Sigma: {round(self.math_sigma, self.rounding_factor)}\n"
                text += "      -Moda - 3 * Sigma: {0}\n".format(round(self.math_wait - 3 * self.math_sigma,
                                                                      self.rounding_factor))
                text += "      -Moda - 2 * Sigma: {0}\n".format(round(self.math_wait - 2 * self.math_sigma,
                                                                      self.rounding_factor))
                text += "      -Moda - 1 * Sigma: {0}\n".format(round(self.math_wait - 1 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 1 * Sigma: {0}\n".format(round(self.math_wait + 1 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 2 * Sigma: {0}\n".format(round(self.math_wait + 2 * self.math_sigma),
                                                                self.rounding_factor)
                text += "      -Moda + 3 * Sigma: {0}\n".format(round(self.math_wait + 3 * self.math_sigma),
                                                                self.rounding_factor)
        return text

    def to_json(self) -> dict:
        """
        This method export DataSet statistics as json
        :return: dict[column name, json statistic]
        """
        data = {"column_name": self.column_name,
                "type": self.field_type,
                "count": self.count,
                "count_unique": self.count_unique,
                "count_NaN": self.nan_count
                }
        if self.field_type.startswith("int") or self.field_type == "float":
            data["numerical_indicators"] = {"min": self.min,
                                            "max": self.max,
                                            "average": self.average
                                            }
            if self.normal_distribution:
                data["normal_distribution"] = {
                    "moda": self.math_moda,
                    "wait": round(self.math_wait, self.rounding_factor),
                    "dispersion": round(self.math_dispersion, self.rounding_factor),
                    "sigma": round(self.math_sigma, self.rounding_factor),
                    "moda-3Xsigma": round(self.math_wait - 3 * self.math_sigma, self.rounding_factor),
                    "moda-2Xsigma": round(self.math_wait - 2 * self.math_sigma, self.rounding_factor),
                    "moda-1Xsigma": round(self.math_wait - 1 * self.math_sigma, self.rounding_factor),
                    "moda+1Xsigma": round(self.math_wait + 1 * self.math_sigma, self.rounding_factor),
                    "moda+2Xsigma": round(self.math_wait + 2 * self.math_sigma, self.rounding_factor),
                    "moda+3Xsigma": round(self.math_wait + 3 * self.math_sigma, self.rounding_factor)

                }
        return {self.column_name: data}

    def _get_nan_count(self) -> int:
        """
        This method calculate count of NaN values
        :return: int
        """
        nan_cnt = 0
        for value in self.values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt

    def _get_math_rasp(self):
        """
        This method calculates the frequency of list values as a percentage
        :return: dict
        """
        math_rasp = {}
        for value in self.values:
            if value in list(math_rasp):
                math_rasp[value] += (1 / len(self.values))
            else:
                math_rasp[value] = (1 / len(self.values))
        return math_rasp

    def _get_math_dispersion(self, math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical variance
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait = self._get_math_wait(math_rasp_dict)
        math_wait_x2 = math_wait * math_wait
        math_wait_2 = self._get_math_wait_2(math_rasp_dict)
        return math_wait_2 - math_wait_x2

    @staticmethod
    def _get_math_moda(math_rasp_dict: dict):
        """
        This method calculates the mathematical mode (the most frequent value)
        :param math_rasp_dict: Dictionary of the frequency of values
        :return:
        """
        moda = -10000000000
        moda_key = -1
        for key in math_rasp_dict:
            if math_rasp_dict[key] > moda:
                moda = math_rasp_dict[key]
                moda_key = key
        return moda_key

    @staticmethod
    def _get_math_wait(math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical expectation
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait = 0
        for md in math_rasp_dict:
            math_wait += float(md) * math_rasp_dict[md]
        return math_wait

    @staticmethod
    def _get_math_wait_2(math_rasp_dict: dict) -> float:
        """
        This method calculates the mathematical expectation squared
        :param math_rasp_dict: Dictionary of the frequency of values
        :return: float
        """
        math_wait_2 = 0
        for md in math_rasp_dict:
            math_wait_2 += float(md) * float(md) * math_rasp_dict[md]
        return math_wait_2


def merge_two_dicts(dict1, dict2):
    """
    This method merge two dicts
    :param dict1: First dict to merge
    :param dict2: Second dict to merge
    :return: result merged dict
    """
    return {**dict1, **dict2}
