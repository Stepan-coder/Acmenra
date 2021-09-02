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
print(original_dataset)
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
task.load_DataFrame(dataframe=original_dataset.get_DataFrame())
task.delete_column(column='SalePrice')
target = original_dataset.get_column(column='SalePrice')
target_analitic = original_dataset.get_column_info(column_name='SalePrice', extended=True)

manager = Manager()
regs = manager.blitz_test_regressions(task=task.get_DataFrame(), target=pd.DataFrame(target), train_split=1200,
                                      prefit=False, n_jobs=-1, show=True)
model = manager.get_model(model_name=regs[0].model_name)
converter_name = regs[0].converter_name
print(len(regs))
quit()
converter = None
converted_task = task.get_DataFrame()
if regs[0].converter_name == "StandardScaler":
    converter = StandardScaler()
elif regs[0].converter_name == "MinMaxScaler":
    converter = MinMaxScaler()
elif regs[0].converter_name == "RobustScaler":
    converter = RobustScaler()
elif regs[0].converter_name == "MaxAbsScaler":
    converter = MaxAbsScaler()
elif regs[0].converter_name == "Normalizer":
    converter = Normalizer()
if converter is not None:
    converted_task = converter.fit_transform(converted_task)
model.set_params(task=pd.DataFrame(converted_task), target=pd.DataFrame(target), train_split=1200, show=True)
# model.fit_grid(count=1, grid_n_jobs=-1)
# locked_params = model.get_grid_locked_params()
model.fit(grid_params=False, n_jobs=1)
print(model)
# Кароче, регрессию он почти решает... Но надо много ещё чего сделать
quit()
answer_dataset = DataSet(dataset_project_name="Answer DataSet", show=True)
answer_dataset.load_csv_dataset(csv_file="OriginalData/test.csv", delimiter=",")
answer_dataset.fillna()
for key in original_dataset.get_keys():
    key_column = original_dataset.get_column_info(column_name=key, extended=True)
    if key in encoders:
        answer_dataset.delete_column(column=key)
        answer_dataset.add_column(column=key, values=encoders[key].encode(key_column.get_values().tolist()))
answer_dataset.update_dataset_info()