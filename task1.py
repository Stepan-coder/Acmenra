from Ra_feature_package.DataSet.DataSet import *
from Ra_feature_package.Preprocessing.Preprocessing import *
from Ra_feature_package.models.Regression.RFRegressor import *

original_dataset = DataSet(dataset_project_name="Original dataset")
original_dataset.load_csv_dataset(csv_file="pizza_v1.csv", delimiter=",")
for i in range(len(original_dataset)):
    original_dataset.set_to_field(column='price_rupiah',
                                  index=i,
                                  value=original_dataset.get_from_field(column='price_rupiah',
                                                                        index=i).replace('Rp', '').replace(',', ''))
original_dataset.set_field_type(field_name='price_rupiah', new_field_type=int)
encoders = {}
for key in original_dataset.get_keys():
    key_column = original_dataset.get_column_info(column_name=key, extended=True)
    if key_column.get_dtype() == "categorical":
        encoders[key] = Encoder()
        encoders[key].fit(key_column.get_values().tolist())
        original_dataset.delete_column(column=key)
        original_dataset.add_column(column=key, values=encoders[key].encode(key_column.get_values().tolist()))
original_dataset.update_dataset_info()
task = DataSet(dataset_project_name='task')
task.load_DataFrame(dataframe=original_dataset.get_dataframe())
task.delete_column(column='price_rupiah')
target = original_dataset.get_column(column='price_rupiah')
target_analitic = original_dataset.get_column_info(column_name='price_rupiah',extended=True)
rfr = RFRegressor(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=100)
rfr.fit(n_jobs=-1, show=True)
print(rfr.get_mean_absolute_error())




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


