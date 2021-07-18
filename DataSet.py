import pandas as pd

# GROUPS['USAGE_AUDIO'] = GROUPS['USAGE_AUDIO'].to_string()
from typing import Dict


class DataSet:
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_folder: str = None,
                 filename: str = None,
                 delimiter: str = None,
                 show: bool = False):
        self.dataset = None  # Dataset in pd.DataFrame
        self.dataset_len = None  # Number of records in the dataset
        self.dataset_keys = None  # Column names in the dataset
        self.dataset_keys_count = None  # Number of columns in the dataset
        self.show = True
        self.load_dataset(dataset=dataset,
                          file_folder=file_folder,
                          filename=filename,
                          delimiter=delimiter)
        # self.get_dataset_analytics()

    def get_dataset_analytics(self):
        """
        This method calculates column metrics
        :return:
        """
        dataset_analytics = {}
        if self.dataset is not None:
            for key in self.dataset_keys:
                if key not in dataset_analytics:
                    dataset_analytics[key] = DataSetFieldAnalytics(key, self.dataset[key])
                    print(dataset_analytics[key])

    def load_dataset(self,
                     dataset: pd.DataFrame = None,
                     file_folder: str = None,
                     filename: str = None,
                     delimiter: str = None,
                     encoding: str = 'utf-8'):
        """
        This method loads the dataset into the DataSet class
        :param dataset: Explicitly specifying pd. DataFrame as a dataset
        :param file_folder: The path to the .csv file
        :param filename: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return:
        """
        if dataset is not None:  # If pd.DataFrame was uploaded to us, we believe that this is the dataset
            self.dataset = dataset
        elif filename is not None and delimiter is not None:  # Otherwise, we load the dataset from csv
            self.dataset = self.read_from_csv(filename=filename,
                                              file_folder=file_folder,
                                              delimiter=delimiter,
                                              encoding=encoding)
            self.dataset.keys()
        else:  # Otherwise, we believe that the dataset will be handed over to us later
            self.dataset = None
        self.update_dataset_base_info()

    def update_dataset_base_info(self):
        """
        This method updates the basic information about the dataset
        """
        if self.dataset is not None:
            self.dataset_len = len(self.dataset)
            self.dataset_keys = self.dataset.keys()
            self.dataset_keys_count = len(self.dataset.keys())

    @staticmethod
    def read_from_csv(filename: str,
                      delimiter: str,
                      file_folder: str = None,
                      encoding: str = 'utf-8'):
        """
        This method reads the dataset from a .csv file
        :param file_folder: The path to the .csv file
        :param filename: The name of the .csv file
        :param delimiter: Symbol-split in a .csv file
        :param encoding: Explicit indication of the .csv file encoding
        :return: pd.DataFrame
        """
        if file_folder is None:  # If the path to the file is specified, then we use it to specify the path to the file
            return pd.read_csv(filename,
                               encoding=encoding,
                               delimiter=delimiter)
        else:
            return pd.read_csv(file_folder + "\\" + filename,
                               encoding=encoding,
                               delimiter=delimiter)

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
                self.dataset[field_name] = self.dataset[field_name].replace(",", ".", regex=True)\
                                                                   .replace(" ", "", regex=True)\
                                                                   .replace(u'\xa0', u'', regex=True)\
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


class DataSetFieldAnalytics:
    def __init__(self,
                 column_name: str,
                 values: list):
        self.column_name = column_name
        self.values = values
        self.field_type = type(values[0]).__name__
        self.count = len(values)
        self.count_unique = len(list(set(values)))
        self.nan_count = self.get_nan_count()
        if self.field_type.startswith("int") or self.field_type == "float":
            self.min = min(values)
            self.max = max(values)
            self.average = sum(values) / self.count

    def __str__(self):
        text = f"DataSetField:\n"
        text += f"  -Name: \'{self.column_name}\'\n"
        text += f"  -Type: \'{self.field_type}\'\n"
        text += f"  -Count: {self.count}\n"
        text += f"  -Count unique: {self.count_unique}\n"
        text += f"  -Count NaN: {self.nan_count}\n"
        if self.field_type.startswith("int") or self.field_type == "float":
            text += f"  Numerical indicators:\n"
            text += f"      -Min: {self.min}\n"
            text += f"      -Max: {self.max}\n"
            text += f"      -Average: {round(self.average, 2)}\n"
        # Допилить Статистические методы, но с "оригинальным" датасетом фильтрацию проводить нельзя, т.к. можно "отрезать" не то

        return text

    def get_nan_count(self):
        nan_cnt = 0
        for value in self.values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt

