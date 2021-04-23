import pandas as pd
from csv_actions import *
from DecisionTreeClassifier import *
from RandomForestClassifier import *
from sklearn.metrics import roc_auc_score

#made_dataset
# train = read_from_csv("made", "train.csv", ",")
# answers = read_from_csv("made", "train-target.csv", ",")
# data = pd.concat([train, answers], axis=1)
# dtc = DTreeClassifier(data_df=data, target="answer", train_split=0.8, show=True)
# rfc = RForestClassifier(data_df=data, target="answer", train_split=0.8, show=True)

#floors_dataset
# num_features = ['agent_fee', 'floor', 'floors_total', 'kitchen_area', 'living_area', 'price', 'rooms_offered', 'total_area', 'total_images', 'exposition_time']
# data = read_from_csv("floors", 'data.tsv', '\t')
# data = data.apply(pd.to_numeric, errors='coerce')
# data = data.dropna()
#
# dtc = DTreeClassifier(data_df=data[num_features], target="exposition_time", train_split=0.5, show=True)
# rfc = RForestClassifier(data_df=data[num_features], target="exposition_time", train_split=0.5, show=True)


#diabets_dataset
data = read_from_csv("diabets", "diabetes.csv", ",")
dtc = DTreeClassifier(data_df=data, target="Outcome", train_split=0.8, show=True)
rfc = RForestClassifier(data_df=data, target="Outcome", train_split=0.8, show=True)

dtc.fit_grid()
dtc.fit(params=dtc.get_best_params())
print("roc_auc: ", dtc.get_roc_auc_score())
print("mean_squared_error: ", dtc.get_mean_squared_error())
print("mean_absolute_error: ", dtc.get_mean_absolute_error())
print("feature_importances: ", dtc.get_feature_importances())
print("best_params: ", dtc.get_best_params())
rfc_params = dtc.get_best_params(is_list=True)
rfc_params['n_estimators'] = [i * 10 for i in range(1, 20 + 1)]
rfc.fit_grid(params_dict=rfc_params)
rfc.fit(params=rfc.get_best_params())
print("roc_auc: ", rfc.get_roc_auc_score())
print("mean_squared_error: ", rfc.get_mean_squared_error())
print("mean_absolute_error: ", rfc.get_mean_absolute_error())
print("feature_importances: ", rfc.get_feature_importances())


