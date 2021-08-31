from Ra_feature_package.Manager.Regression import *


class Manager:
    def __init__(self):
        self.regression = Regression()

    def get_models_names(self):
        return self.regression.get_models_names()

    def get_model(self, model_name: str):
        if model_name in self.regression.get_models_names():
            return self.regression.get_model(model_name=model_name)

    def get_prefit_model_locked_params(self, model):
        pass

    def blitz_test_regressions(self,
                               task: pd.DataFrame,
                               target: pd.DataFrame,
                               train_split: int,
                               prefit: bool = False,
                               n_jobs: int = 1,
                               show: bool = False):
        return self.regression.blitz_test(task=task,
                                          target=target,
                                          train_split=train_split,
                                          prefit=prefit,
                                          n_jobs=n_jobs,
                                          show=show)
