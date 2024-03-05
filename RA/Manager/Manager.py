from prettytable import PrettyTable

from RA.DataSet.DataSet import *
from RA.Manager.Regression import *
from RA.Manager.Classification import *


class Manager:
    def __init__(self, path: str, project_name: str) -> None:
        """
        This method initializes the manager's work
        :param path: The path to the project folder
        :param project_name: The name of this project
        """
        self.set_project_name(project_name)
        self.__path = path
        self.__project_path = self.__create_project_folder(path=path)
        self.__datasets = {}
        self.__models = {}
        self.__regression = Regression()
        self.__classification = Classification()

    def __str__(self):
        table = PrettyTable()
        is_dataset = True if len(self.__datasets) + len(self.__models) > 0 else False
        table.title = f"{'Empty ' if not is_dataset else ''}Manager \"{self.__project_name}\""
        table.field_names = ["Name", "Type", "Status"]
        for dataset in self.__datasets:
            status = "Ups.."
            if self.DataSet(dataset).status == DataSetStatus.CREATED:
                status = Manager.__set_str_cmd_clr(self.DataSet(dataset).status.value, 'RED')
            elif self.DataSet(dataset).status == DataSetStatus.EMPTY:
                status = Manager.__set_str_cmd_clr(self.DataSet(dataset).status.value, 'YELLOW')
            elif self.DataSet(dataset).status == DataSetStatus.ENABLE:
                status = Manager.__set_str_cmd_clr(self.DataSet(dataset).status.value, 'GREEN')
            table.add_row([dataset, "<DataSet>", status])
        return str(table)

    def __instancecheck__(self, instance: Any) -> bool:
        """
        This method checks is instance type is Manager
        :param instance: Checked value
        """
        return isinstance(instance, type(self))

    def __len__(self) -> int:
        return len(self.__datasets) + len(self.__models)

    def __create_project_folder(self, path: str) -> str:
        """
        Creates a folder where all project files will be "stacked"
        :param path: The path to the project folder
        """
        if not isinstance(path, str):
            raise Exception("The path must be a string!")
        if len(path) == 0:
            raise Exception("The 'path' must be longer than 0 characters!")
        if not os.path.exists(path):
            raise Exception("Wrong way!")
        if not os.path.exists(os.path.join(path, self.__project_name)):
            os.makedirs(os.path.join(path, self.__project_name))
        return os.path.join(path, self.__project_name)

    def set_project_name(self, new_project_name: str) -> None:
        """
        This method sets the new name of this project
        :param new_project_name: The new name of this project
        """
        if not isinstance(new_project_name, str):
            raise Exception("The new_project_name must be a string!")
        if len(new_project_name) == 0:
            raise Exception("The 'new_project_name' must be longer than 0 characters!")
        self.__project_name = new_project_name

    def get_project_name(self) -> str:
        """
        This method return the name of this project
        """
        return self.__project_name

    def set_path(self, new_path: str) -> None:
        """
        This method sets a new path of this project
        :param new_path: new path of project
        """
        self.__project_path = self.__create_project_folder(path=new_path)

    def get_path(self) -> str:
        """
        This method returns the project path
        """
        return self.__project_path

    def DataSet(self, dataset_name: str) -> DataSet:
        """
        This method returns DataSet from manager
        :param dataset_name: The DataSet class name
        """
        if dataset_name not in self.__datasets:
            raise Exception("There is no DataSet with this name!")
        return self.__datasets[str(dataset_name)]

    def create_DataSet(self, dataset_name: str) -> None:
        """
        This method creates a new DataSet in manager
        :param dataset_name: The name of new DataSet
        """
        if dataset_name in self.__datasets:
            raise Exception("A dataset with this name already exists!")
        self.__datasets[dataset_name] = DataSet(dataset_name=dataset_name)
        self.__datasets[dataset_name].set_saving_path(path=self.__project_path)

    def add_DataSet(self, dataset: DataSet) -> None:
        """
        This method adds a DataSet to the manager
        :param dataset: The DataSet class that we want to add
        """
        if str(dataset.get_name()) in self.__datasets:
            raise Exception("A dataset with this name already exists!")
        self.__datasets[dataset.get_name()] = dataset
        self.__datasets[dataset.get_name()].set_saving_path(path=self.__project_path)

    def delate_DataSet(self, dataset_name: str) -> None:
        """
        This method delete a DataSet from manager
        :param dataset_name: The DataSet class that we want delete
        """
        if dataset_name not in self.__datasets:
            raise Exception("There is no dataset with this name!")
        del self.__datasets[str(dataset_name)]

    def split_DataSet(self, dataset_name: str, count: int, delete_original_DataSet: bool = False) -> List[str]:
        """
        This method splits the dataset into datasets of the specified length and adds them to the project
        :param dataset_name: The DataSet class name
        :param count: Maximal count of rows in new DataSets
        :param delete_original_DataSet: This switch is responsible for auto-deleting the original dataset
        after splitting it
        """
        if dataset_name not in self.__datasets:
            raise Exception("There is no DataSet with this name!")
        splitted_datasets = self.__datasets[str(dataset_name)].split(count=count)
        for i in range(len(splitted_datasets)):
            if splitted_datasets[i].get_name() in self.__datasets:
                raise Exception(f"As a result of splitting the dataset \'{dataset_name}\' into parts, "
                                f"the name of the dataset {splitted_datasets[i].get_name()} coincided!")
        splitted_datasets_names = []
        for i in range(len(splitted_datasets)):
            self.__datasets[splitted_datasets[i].get_name()] = splitted_datasets[i]
            splitted_datasets_names.append(splitted_datasets[i].get_name())
        if delete_original_DataSet:
            del self.__datasets[str(dataset_name)]
        return splitted_datasets_names

    def concat_DataSets(self,
                        new_dataset_name: str,
                        dataset_names: List[str]=None,
                        only_new_dataset=True) -> None:
        """
        This method is the reverse of "split_DataSet".
        It connects all the manager's DataSets sequentially, into a new, large DataSet
        :param only_new_dataset:
        """
        if len(self.__datasets) == 0:
            raise Exception("No dataset has been added to the manager yet!")
        if len(self.__datasets) == 1:
            warnings.warn("The manager has only 1 available dataset!")
            return None
        if dataset_names is None:
            datasets = list(self.__datasets.keys())
        else:
            datasets = dataset_names
        ds_keys = []
        for ds in datasets:
            if ds not in self.__datasets:
                raise Exception("There is no DataSet with this name!")
            if set(self.DataSet(ds).get_keys()) in ds_keys or len(ds_keys) == 0:
                ds_keys.append(set(self.DataSet(ds).get_keys()))
            else:
                raise Exception("The names of the columns in the datasets do not match!")
        self.create_DataSet(dataset_name=new_dataset_name)
        for ds in datasets:
            self.DataSet(new_dataset_name).concat_DataSet(dataset=self.DataSet(dataset_name=ds))
            if only_new_dataset:
                self.delate_DataSet(dataset_name=ds)

    def get_DataSets(self) -> Dict[str, DataSet]:
        """
        This method returns all DataSets from manager
        """
        return self.__datasets

    def get_datasets_names(self) -> List[str]:
        """
        This method returns the names of all models added to the RA system
        """
        return list(self.__datasets.keys())

    def get_models_names(self) -> List[str]:
        """
        This method returns the names of all models added to the RA system
        """
        return self.__regression.get_models_names() + self.__classification.get_models_names()

    def Model(self, model_name: str):
        """

        :param model_name: Model Name
        """
        if model_name in self.__regression.get_models_names():
            return self.__regression.get_model(model_name=model_name)
        elif model_name in self.__classification.get_models_names():
            return self.__classification.get_model(model_name=model_name)
        else:
            raise Exception(f"The \'{abs}\' model does not exist!")

    def blitz_test_regressions(self,
                               task: pd.DataFrame,
                               target: pd.DataFrame,
                               train_split: int,
                               prefit: bool = False,
                               n_jobs: int = 1,
                               show: bool = False):
        """
        Performs auto-testing of models and sorts them in descending order of accuracy
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param prefit: Responsible for pre-training models based on blocked parameters
        :param n_jobs: The number of jobs to run in parallel.
        :param show: The parameter responsible for displaying the progress of work
        """
        return self.__regression.blitz_test(task=task,
                                            target=target,
                                            train_split=train_split,
                                            prefit=prefit,
                                            n_jobs=n_jobs,
                                            show=show)

    def blitz_test_classification(self,
                                  task: pd.DataFrame,
                                  target: pd.DataFrame,
                                  train_split: int,
                                  prefit: bool = False,
                                  n_jobs: int = 1,
                                  show: bool = False):
        """
        Performs auto-testing of models and sorts them in descending order of accuracy
        :param task: The training part of the dataset
        :param target: The target part of the dataset
        :param train_split: The coefficient of splitting into training and training samples
        :param prefit: Responsible for pre-training models based on blocked parameters
        :param n_jobs: The number of jobs to run in parallel.
        :param show: The parameter responsible for displaying the progress of work
        """
        return self.__classification.blitz_test(task=task,
                                                target=target,
                                                train_split=train_split,
                                                prefit=prefit,
                                                n_jobs=n_jobs,
                                                show=show)

    @staticmethod
    def __set_str_cmd_clr(text: str, color: str) -> str:
        clr_text = ""
        if color == 'RED':
            clr_text = "\033[31m {}\033[0m".format(text)
        elif color == 'GREEN':
            clr_text = "\033[32m {}\033[0m".format(text)
        elif color == 'YELLOW':
            clr_text = "\033[33m {}\033[0m".format(text)
        return clr_text
