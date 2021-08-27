import pandas as pd

from Ra_feature_package.Manager.Manager import *
from Ra_feature_package.DataSet.DataSet import *
from Ra_feature_package.Preprocessing.Preprocessing import *


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
print(original_dataset)
task = DataSet(dataset_project_name='task')
task.load_DataFrame(dataframe=original_dataset.get_dataframe())
task.delete_column(column='price_rupiah')
target = original_dataset.get_column(column='price_rupiah')
target_analitic = original_dataset.get_column_info(column_name='price_rupiah', extended=True)


# blitz_test_regressor(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=100, show=True)
# quit()

rfr = LinSVRegressor(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=100, show=True)
rfr.fit_grid(count=0,
             grid_n_jobs=-1)
params = rfr.get_grid_best_params()
print(params)
rfr.fit(grid_params=True, n_jobs=-1)
print(rfr)
rfr.fit(grid_params=True, n_jobs=-1)
print(rfr)
rfr1 = LinSVRegressor(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=100, show=True)
rfr1.fit(param_dict=params, n_jobs=-1)
print(rfr1)
