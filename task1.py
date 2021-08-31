import warnings
import pandas as pd
from Ra_feature_package.Manager.Manager import *
from Ra_feature_package.Manager.Regression import *
from Ra_feature_package.DataSet.DataSet import *
from Ra_feature_package.Preprocessing.Preprocessing import *


warnings.filterwarnings("ignore")

original_dataset = DataSet(dataset_project_name="Original DataSet", show=True)
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

manager = Manager()
regs = manager.blitz_test_regressions(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=1200,
                                      prefit=False, n_jobs=-1, show=True)
model = manager.get_model(model_name=regs[0].model_name)
model.set_params(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=1200, show=True)
model.fit_grid(count=0, grid_n_jobs=-1)
model.fit(grid_params=True, n_jobs=-1)
best_params = model.get_grid_best_params()
locked_params = {}
for bp in best_params:
    if bp in model.get_locked_params():
        locked_params[bp] = best_params[bp]
print(model)
# Кароче, регрессию он почти решает... Но надо много ещё чего сделать
quit()
