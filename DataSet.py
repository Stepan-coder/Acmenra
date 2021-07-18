import pandas as pd


class DataSet:
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 file_folder: str = None,
                 filename: str = None,
                 delimiter: str = None):
        self.dataset = None  # Dataset Object
        self.dataset_len = None  # Number of records in the dataset
        self.dataset_keys = None  # Column names in the dataset
        self.dataset_keys_count = None  # Number of columns in the dataset

        self.load_dataset(dataset=dataset,
                          file_folder=file_folder,
                          filename=filename,
                          delimiter=delimiter)

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
