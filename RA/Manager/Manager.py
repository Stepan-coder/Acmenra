from RA.Manager.Regression import *
from RA.Manager.Classification import *


class Manager:
    def __init__(self):
        self.regression = Regression()
        self.classification = Classification()

    def get_models_names(self) -> List[str]:
        """
        This method returns the names of all models added to the RA system
        :return: Names of models
        """
        return self.regression.get_models_names() + self.classification.get_models_names()

    def get_model(self, model_name: str):
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

    def get_prefit_model_locked_params(self, model):
        pass

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