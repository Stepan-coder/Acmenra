from sklearn.metrics import roc_auc_score
from Ra_feature_package.DataSet.DataSet import *

original_dataset = DataSet(dataset_project_name="Original dataset")
original_dataset.load_csv_dataset(csv_file="pizza_v1.csv",
                                  delimiter=",")
print(original_dataset.head())
for i in range(len(original_dataset)):
    original_dataset.set_to_field(column='price_rupiah',
                                  index=i,
                                  value=original_dataset.get_from_field(column='price_rupiah',
                                                                        index=i).replace('Rp', '').replace(',', ''))
print(original_dataset.head())
print(original_dataset.get_column_info(column_name='price_rupiah').get_count())


# dtc.fit_grid()
# dtc.fit(params=dtc.get_best_params())
# print("roc_auc: ", dtc.get_roc_auc_score())
# print("mean_squared_error: ", dtc.get_mean_squared_error())
# print("mean_absolute_error: ", dtc.get_mean_absolute_error())
# print("feature_importances: ", dtc.get_feature_importances())
# print("best_params: ", dtc.get_best_params())
# rfc_params = dtc.get_best_params(is_list=True)
# rfc_params['n_estimators'] = [i * 10 for i in range(1, 20 + 1)]
# rfc.fit_grid(params_dict=rfc_params)
# rfc.fit(params=rfc.get_best_params())
# print("roc_auc: ", rfc.get_roc_auc_score())
# print("mean_squared_error: ", rfc.get_mean_squared_error())
# print("mean_absolute_error: ", rfc.get_mean_absolute_error())
# print("feature_importances: ", rfc.get_feature_importances())


