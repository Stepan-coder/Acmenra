import os
import warnings
import pandas as pd
from RA.Manager.Manager import *
from RA.Manager.Regression import *
from RA.DataSet.DataSet import *
from RA.Preprocessing.LabelEncoder import *


warnings.filterwarnings("ignore")

manager = Manager(path=os.getcwd(), project_name="test_project")
manager.add_DataSet(dataset=DataSet(dataset_name="Original DataSet", show=True))
manager.DataSet("Original DataSet").create_empty_dataset(columns_names=["A", "B"])
manager.DataSet("Original DataSet").add_row(new_row={"A": 1, "B": 1})
manager.DataSet("Original DataSet").add_row(new_row={"A": 2, "B": 2})
manager.DataSet("Original DataSet").add_row(new_row={"A": 3, "B": 3})
manager.DataSet("Original DataSet").add_row(new_row={"A": 4, "B": 1})
manager.DataSet("Original DataSet").add_row(new_row={"A": 5, "B": 2})
manager.DataSet("Original DataSet").add_row(new_row={"A": 6, "B": 3})
manager.DataSet("Original DataSet").add_row(new_row={"A": 7, "B": 1})
manager.DataSet("Original DataSet").add_row(new_row={"A": 8, "B": 2})
manager.DataSet("Original DataSet").add_row(new_row={"A": 9, "B": 3})
manager.DataSet("Original DataSet").add_row(new_row={"A": 10, "B": 3})
# manager.DataSet("Original DataSet").head()
splitted_DatatSet = manager.DataSet("Original DataSet").split(count=3)
splitted_DatatSet[0].head()
print()
splitted_DatatSet[1].head()
print()
splitted_DatatSet[2].head()
print()
splitted_DatatSet[3].head()
for sp in splitted_DatatSet:
    print(len(sp))
# manager.add_DataSet(dataset=DataSet(dataset_name="Original DataSet1", show=True))
# manager.DataSet("Original DataSet1").create_empty_dataset(columns_names=["A", "B"])
# manager.DataSet("Original DataSet1").add_row(new_row={"A": 4, "B": 4})
# manager.DataSet("Original DataSet1").add_row(new_row={"A": 5, "B": 5})
# manager.DataSet("Original DataSet1").add_row(new_row={"A": 6, "B": 6})
# manager.DataSet("Original DataSet1").head()
# manager.DataSet("Original DataSet").concat_DataSet(dataset=manager.DataSet("Original DataSet1"))
# manager.DataSet("Original DataSet").head(10)
# quit()
# manager.DataSet(dataset_name="Original DataSet").load_csv_dataset(csv_file="OriginalData/train.csv", delimiter=",")
# manager.DataSet(dataset_name="Original DataSet").fillna()
# manager.DataSet(dataset_name="Original DataSet").export(dataset_name="123")


#
# encoders = {}
# for key in original_dataset.get_keys():
#     key_column = original_dataset.get_column_info(column_name=key, extended=True)
#     if key_column.get_dtype(50) == "categorical":
#         encoders[key] = LabelEncoder()
#         encoders[key].fit(key_column.get_values().tolist())
#         original_dataset.delete_column(column=key)
#         original_dataset.add_column(column=key, values=encoders[key].encode(key_column.get_values().tolist()))
# original_dataset.update_dataset_info()
#
# task = DataSet(dataset_name='task')
# task.load_DataFrame(dataframe=manager.DataSet(dataset_name="Original DataSet").get_dataframe())
# task.delete_column(column='SalePrice')
# target = manager.DataSet(dataset_name="Original DataSet").get_column(column='SalePrice')
# target_analitic = manager.DataSet(dataset_name="Original DataSet").get_column_info(column_name='SalePrice',
#                                                                                    extended=True)
#
# regs = manager.blitz_test_regressions(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=1200,
#                                       prefit=True, n_jobs=-1, show=True)
# model = manager.Model(model_name=regs[0].model_name)
# quit()
# converter = None
# converted_task = task.get_dataframe()
# if regs[0].converter_name == "StandardScaler":
#     converter = StandardScaler()
# elif regs[0].converter_name == "MinMaxScaler":
#     converter = MinMaxScaler()
# elif regs[0].converter_name == "RobustScaler":
#     converter = RobustScaler()
# elif regs[0].converter_name == "MaxAbsScaler":
#     converter = MaxAbsScaler()
# elif regs[0].converter_name == "Normalizer":
#     converter = Normalizer()
# if converter is not None:
#     converted_task = converter.fit_transform(converted_task)
# model.set_params(task=pd.DataFrame(converted_task), target=pd.DataFrame(target), train_split=1200, show=True)
# # model.fit_grid(count=1, grid_n_jobs=-1)
# # locked_params = model.get_grid_locked_params()
# model.fit(grid_params=False, n_jobs=1)
# print(model)
# quit()
# answer_dataset = DataSet(dataset_project_name="Answer DataSet", show=True)
# answer_dataset.load_csv_dataset(csv_file="OriginalData/test.csv", delimiter=",")
# answer_dataset.fillna()
# for key in original_dataset.get_keys():
#     key_column = original_dataset.get_column_info(column_name=key, extended=True)
#     if key in encoders:
#         answer_dataset.delete_column(column=key)
#         answer_dataset.add_column(column=key, values=encoders[key].encode(key_column.get_values().tolist()))
# answer_dataset.update_dataset_info()