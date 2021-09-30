import os

from RA.DataSet.DataSet import *
from RA.Manager.Regression import *
from RA.Manager.Classification import *


class Manager:
    def __init__(self, path: str, project_name: str):
        """

        :param path:
        :param project_name:
        """
        self.project_name = project_name
        self.project_path = self.__create_project_folder(path=path)
        self.datasets = {}
        self.regression = Regression()
        self.classification = Classification()

    def __create_project_folder(self, path):
        if not os.path.exists(path):
            raise Exception("Wrong way!")
        if not os.path.exists(os.path.join(path, self.project_name)):
            os.makedirs(os.path.join(path, self.project_name))
        return os.path.join(path, self.project_name)

    # def cmd(self, cmd: str):  # Метод для доступа к функционалу из строки с командой
    #     if cmd.startswith("LOAD DATASET"):

    def add_DataSet(self, dataset: DataSet) -> None:
        """
        This method adds a DataSet to the manager
        :param dataset: The DataSet class that we want to add
        :return: None
        """
        if str(dataset.get_name()) not in self.datasets:
            self.datasets[dataset.get_name()] = dataset
            self.datasets[dataset.get_name()].set_saving_path(path=self.project_path)
        else:
            raise Exception("A dataset with this name already exists!")

    def DataSet(self, dataset_name: str) -> DataSet:
        """
        This method returns DataSet from manager
        :param dataset_name: The DataSet class name
        :return: DataSet
        """
        if dataset_name not in self.datasets:
            raise Exception("There is no dataset with this name!")
        return self.datasets[str(dataset_name)]

    def delate_DataSet(self, dataset_name: str) -> None:
        """
        This method delete a DataSet from manager
        :param dataset_name: The DataSet class that we want delete
        :return: None
        """
        if dataset_name not in self.datasets:
            raise Exception("There is no dataset with this name!")
        del self.datasets[str(dataset_name)]

    def get_models_names(self) -> List[str]:
        """
        This method returns the names of all models added to the RA system
        :return: Names of models
        """
        return self.regression.get_models_names() + self.classification.get_models_names()

    def Model(self, model_name: str):
        """

        :param model_name: Model Name
        :return: RA.Model
        """
        if model_name in self.regression.get_models_names():
            return self.regression.get_model(model_name=model_name)
        elif model_name in self.classification.get_models_names():
            return self.classification.get_model(model_name=model_name)
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
        :return: list of models
        """
        return self.regression.blitz_test(task=task,
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
        :return: list of models
        """
        return self.classification.blitz_test(task=task,
                                              target=target,
                                              train_split=train_split,
                                              prefit=prefit,
                                              n_jobs=n_jobs,
                                              show=show)