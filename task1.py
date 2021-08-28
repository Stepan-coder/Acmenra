import warnings
import pandas as pd
from Ra_feature_package.Manager.Manager import *
from Ra_feature_package.DataSet.DataSet import *
from Ra_feature_package.Preprocessing.Preprocessing import *


warnings.filterwarnings("ignore")

original_dataset = DataSet(dataset_project_name="Original DataSet", show=True, )
original_dataset.load_csv_dataset(csv_file="OriginalData/train.csv", delimiter=",")
original_dataset.fillna()
encoders = {}
for key in original_dataset.get_keys():
    key_column = original_dataset.get_column_info(column_name=key, extended=True)
    if key_column.get_dtype(50) == "categorical":
        encoders[key] = Encoder()
        encoders[key].fit(key_column.get_values().tolist())
        original_dataset.delete_column(column=key)
        original_dataset.add_column(column=key, values=encoders[key].encode(key_column.get_values().tolist()))
original_dataset.update_dataset_info()

task = DataSet(dataset_project_name='task')
task.load_DataFrame(dataframe=original_dataset.get_dataframe())
task.delete_column(column='SalePrice')
target = original_dataset.get_column(column='SalePrice')
target_analitic = original_dataset.get_column_info(column_name='SalePrice', extended=True)

print(blitz_test_regressor(task=task.get_dataframe(), target=pd.DataFrame(target),
                           train_split=100, show=True,
                           prefit=True,
                           n_jobs=-1))
quit()
