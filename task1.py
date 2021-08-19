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
print(original_dataset)
task = DataSet(dataset_project_name='task')
task.load_DataFrame(dataframe=original_dataset.get_dataframe())
task.delete_column(column='price_rupiah')
target = original_dataset.get_column(column='price_rupiah')
target_analitic = original_dataset.get_column_info(column_name='price_rupiah', extended=True)
rfr = RFRegressor(task=task.get_dataframe(), target=pd.DataFrame(target), train_split=100, show=True)
rfr.fit_grid(count=5, grid_n_jobs=6)
rfr.fit(grid_params=True, n_jobs=-1)
# print(rfr.get_mean_absolute_error())
# rfr.get_predict_test_plt(show=True)
print(rfr)



# "   -Param 'n_estimators'(1): [100]
#    -Param 'criterion'(2): ['mse', 'mae']
#    -Param 'max_depth'(1): [None]
#    -Param 'min_samples_split'(1): [2]
#    -Param 'min_samples_leaf'(1): [1]
#    -Param 'min_weight_fraction_leaf'(1): [0.0]
#    -Param 'max_features'(4): ['sqrt', 'auto', 'log2', None]
#    -Param 'max_leaf_nodes'(1): [None]
#    -Param 'min_impurity_decrease'(1): [0.0]
#    -Param 'bootstrap'(2): [True, False]
#    -Param 'oob_score'(1): [False]
#    -Param 'verbose'(1): [0]
#    -Param 'warm_start'(2): [True, False]
#    -Param 'ccp_alpha'(1): [0.0]
#    -Param 'max_samples'(1): [None]"

# +--------------------------------------------+
# |       "RandomForestRegressor" model        |
# +-----------------------+--------------------+
# |         Error         |       Result       |
# +-----------------------+--------------------+
# |     ROC AUC score     |        inf         |
# |    R-Squared_error    | 0.9332411200753861 |
# |  Mean Absolute Error  | 6776.551724137931  |
# |   Mean Squared Error  | 73119270.68965517  |
# | Median Absolute Error |       5300.0       |
# +-----------------------+--------------------+