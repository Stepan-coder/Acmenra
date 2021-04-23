import pandas as pd
from csv_actions import *
from DecisionTreeClassifier import *
from RandomForest import *
from sklearn.metrics import roc_auc_score

train = read_from_csv("data", "train.csv", ",")
answers = read_from_csv("data", "train-target.csv", ",")
data = pd.concat([train, answers], axis=1)
dtc = DTreeClassifier(data_df=data, target="answer", train_split=0.8, step=9)
rfc = RForestClassifier(data_df=data, target="answer", train_split=0.8, step=9)
dtc.fit()
print(dtc.get_roc_auc_score())
print(dtc.get_mean_squared_error())
print(dtc.get_feature_importances())
rfc.fit()
print(rfc.get_roc_auc_score())
print(rfc.get_mean_squared_error())
print(rfc.get_feature_importances())


