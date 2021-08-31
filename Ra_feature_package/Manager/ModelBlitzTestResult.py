
class ModelBlitzTestResult:
    def __init__(self, blitz_test):
        self.model_name = blitz_test[0]
        self.converter_name = blitz_test[1]
        self.roc_auc_score = blitz_test[2]
        self.r_squared_error = blitz_test[3]
        self.mean_absolute_error = blitz_test[4]
        self.mean_squared_error = blitz_test[5]
        self.root_mean_squared_error = blitz_test[6]
        self.median_absolute_error = blitz_test[7]