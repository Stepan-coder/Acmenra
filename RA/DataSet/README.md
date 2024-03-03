<details>
  <summary><strong>class DataSet</strong></summary>
  <ul>
    <li><code>func</code> <strong>__init__</strong> - This is an init method</li>
    <li><font color="477FEF"><strong>func</strong></font> <code>__str__</code> - </li>
    <li><code>func</code> <strong>__iter__</strong> - This method allows you to iterate over a data set in a loop. I.e. makes it iterative</li>
    <li><code>func</code> <strong>__reversed__</strong> - This method return a reversed copy of self-class</li>
    <li><code>func</code> <strong>__instancecheck__</strong> - This method checks is instance type is DataSet</li>
    <li><code>func</code> <strong>__len__</strong> - This method returns count rows in this dataset</li>
    <li><code>property</code> <strong>name</strong> - This property returns the dataset name of the current DataSet</li>
    <li><code>property</code> <strong>status</strong> - </li>
    <li><code>property</code> <strong>is_loaded</strong> - This property returns the current state of this DataSet</li>
    <!-- Добавьте остальные элементы списка с необходимыми выделениями -->
  </ul>
</details>

<ul>
  <li><code>func</code> <strong>merge_two_dicts</strong> - This method merge two dicts</li>
</ul>

<details><summary> class DataSet </summary>
* `func` __init__ - This is an init method<br>
* `func` __str__ - 
* `func` __iter__ - This method allows you to iterate over a data set in a loop. I.e. makes it iterative
* `func` __reversed__ - This method return a reversed copy of self-class
* `func` __instancecheck__ - This method checks is instance type is DataSet
* `func` __len__ - This method returns count rows in this dataset
* `property ` name - This property returns the dataset name of the current DataSet
* `property ` status - 
* `property `  is_loaded - This property returns the current state of this DataSet
* `property ` delimiter - This property returns a delimiter character
* `property ` encoding - This property returns the encoding of the current dataset file
* `property ` columns_name - This property return column names of dataset pd.DataFrame
* `property ` columns_count - This method return count of column names of dataset pd.DataFrame
* `property ` supported_formats - This property returns a list of supported files
* `func` head - This method prints the first n rows
* `func` tail - This method prints the last n rows
* `func` set_name - This method sets the project_name of the DataSet
* `func` set_saving_path - This method removes the column from the dataset
* `func` set_delimiter - This method sets the delimiter character
* `func` set_encoding - This method sets the encoding for the future export of the dataset
* `func` set_to_field - This method gets the value from the dataset cell
* `func` get_from_field - This method gets the value from the dataset cell
* `func` add_row - This method adds a new row to the dataset
* `func` get_row - This method returns a row of the dataset in dictionary format, where the keys are the column names and the values are the values in the columns
* `func` delete_row - This method delete row from dataset
* `func` Column - This method summarizes the values from the columns of the dataset and returns them as a list of tuples
* `func` add_column - This method adds the column to the dataset on the right
* `func` get_column - This method summarizes the values from the columns of the dataset and returns them as a list of tuples
* `func` rename_column - This method renames the column in the dataset
* `func` delete_column - This method removes the column from the dataset
* `func` set_columns_types - This method converts column types
* `func` set_column_type - This method converts column type
* `func` get_column_stat - This method returns statistical analytics for a given column
* `func` reverse - This method expands the order of rows in the dataset
* `func` fillna - This method automatically fills in "null" values: for "int" -> 0, for "float" -> 0.0, for "str" -> "-".
* `func` equals -
* `func` diff - 
* `func` split - This method automatically divides the DataSet into a list of DataSets with a maximum of "count" rows in each
* `func` sort_by_column - This method sorts the dataset by column "column_name"
* `func` get_correlations - This method calculate correlations between columns
* `func` get_DataFrame - This method return dataset as pd.DataFrame
* `func` join_DataFrame - This method attaches a new dataset to the current one (at right)
* `func` concat_DataFrame - This method attaches a new dataset to the current one (at bottom)
* `func` concat_DataSet - This method attaches a new dataset to the current one (at bottom)
* `func` update_dataset_info - This method updates, the analitic-statistics data about already precalculated columns
* `func` create_empty_dataset - This method creates an empty dataset
* `func` create_dataset_from_list - This method creates a dataset from list of columns values
* `func` load_DataFrame - This method loads the dataset into the DataSet class
* `func` load_csv_dataset - This method loads the dataset into the DataSet class
* `func` load_excel_dataset - This method loads the dataset into the DataSet class
* `func` load_dataset_project - This method loads the dataset into the DataSet class
* `func` export - This method exports the dataset as DataSet Project
* `func` to_csv - This method saves pd.DataFrame to .csv file
* `func` to_excel - This method saves pd.DataFrame to excel file
* `func` __get_column_type - This method learns the column type
* `func` __read_dataset_info_from_json - This method reads config and statistics info from .json file
* `func` __update_dataset_base_info - This method updates the basic information about the dataset
* `static` __dif_lists_index - 
* `static` __read_from_csv - 
* `static` __read_from_xlsx - 
* `static` get_excel_sheet_names -
</details>

* `func` merge_two_dicts - This method merge two dicts
