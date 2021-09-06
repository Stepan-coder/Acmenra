from Ra_feature_package.DataSet.DataSet import *


original_dataset = DataSet(dataset_project_name="Original DataSet", show=True)
original_dataset.load_csv_dataset(csv_file="OriginalData/train.csv", delimiter=",")
original_dataset.fillna()
print(original_dataset)


this_dataset = DataSet(dataset_project_name="test", show=True)
this_dataset.create_empty_dataset()
this_dataset.add_column(column="LotArea",
                        values=original_dataset.get_column(column="LotArea"))
this_dataset.add_column(column="LotFrontage",
                        values=original_dataset.get_column(column="LotFrontage"))
this_dataset.export(dataset_name="test",
                    dataset_folder="test",
                    delimeter=";")
print(this_dataset)

exel_dataset = DataSet(dataset_project_name="excel")
exel_dataset.load_xlsx_dataset(xlsx_file="DataSet_EKB_200000.xlsx",
                               sheet_name="200000ste")
exel_dataset.fillna()
print(exel_dataset)