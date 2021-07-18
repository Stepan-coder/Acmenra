import pandas as pd

# GROUPS['USAGE_AUDIO'] = GROUPS['USAGE_AUDIO'].to_string()
class DataSet:
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_folder: str = None,
                 filename: str = None,
                 delimiter: str = None):
        self.dataset = None  # Dataset in pd.DataFrame
        self.dataset_len = None  # Number of records in the dataset
        self.dataset_keys = None  # Column names in the dataset
        self.dataset_keys_count = None  # Number of columns in the dataset

        self.load_dataset(dataset=dataset,
                          file_folder=file_folder,
                          filename=filename,
                          delimiter=delimiter)
        self.get_dataset_analytics()

    def get_dataset_analytics(self, show: bool = False):
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
        :return:
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


class DataSetFieldAnalytics:
    def __init__(self,
                 column_name: str,
                 values: list,
                 ):
        self.column_name = column_name
        self.values = values
        self.field_type = type(values[0])
        self.count = len(values)
        self.count_unique = len(list(set(values)))
        self.nan_count = self.get_nan_count()
        # print(isinstance(values[0], str))

        if isinstance(values[0], int) or isinstance(values[0], float):
            self.min = min(values)
            self.max = max(values)
            self.average = sum(values) / self.count

    def __str__(self):
        text = f"DataSetField:\n"
        text += f"*Name: \'{self.column_name}\'\n"
        text += f"*Type: \'{self.field_type}\'\n"
        text += f"*Count(Unique)[NaN values]: {self.count}({self.count_unique})[{self.nan_count}]\n"
        if isinstance(self.values[0], int) or isinstance(self.values[0], float):
            text += f"*Min/Average/Max: {self.min}/{self.average}/{self.max}\n"
        return text

    def get_nan_count(self):
        nan_cnt = 0
        for value in self.values:
            if pd.isna(value):
                nan_cnt += 1
        return nan_cnt

