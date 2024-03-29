<details>
  <summary> class DataSet </summary>
  <ul>
    <li><strong>func</strong> <code>__init__</code> - This is an init method<br></li>
    <li><strong>func</strong> <code>__str__</code> - </li>
    <li><strong>func</strong> <code>__iter__</code> - This method allows you to iterate over a data set in a loop. I.e. makes it iterative</li>
    <li><strong>func</strong> <code>__reversed__</code> - This method return a reversed copy of self-class</li>
    <li><strong>func</strong> <code>__instancecheck__</code> - This method checks is instance type is DataSet</li>
    <li><strong>func</strong> <code>__len__</code> - This method returns count rows in this dataset</li>
    <li><strong>property</strong> <code>name</code> - This property returns the dataset name of the current DataSet</li>
    <li><strong>property</strong> <code>status</code> - </li>
    <li><strong>property</strong> <code>is_loaded</code> - This property returns the current state of this DataSet</li>
    <li><strong>property</strong> <code>delimiter</code> - This property returns a delimiter character</li>
    <li><strong>property</strong> <code>encoding</code> - This property returns the encoding of the current dataset file</li>
    <li><strong>property</strong> <code>columns_name</code> - This property return column names of dataset pd.DataFrame</li>
    <li><strong>property</strong> <code>columns_count</code> - This method return count of column names of dataset pd.DataFrame</li>
    <li><strong>property</strong> <code>supported_formats</code> - This property returns a list of supported files</li>
    <li><strong>func</strong> <code>head</code> - This method prints the first n rows</li>
    <li><strong>func</strong> <code>tail</code> - This method prints the last n rows</li>
    <li><strong>func</strong> <code>set_name</code> - This method sets the project_name of the DataSet</li>
    <li><strong>func</strong> <code>set_saving_path</code> - This method removes the column from the dataset</li>
    <li><strong>func</strong> <code>set_delimiter</code> - This method sets the delimiter character</li>
    <li><strong>func</strong> <code>set_encoding</code> - This method sets the encoding for the future export of the dataset</li>
    <li><strong>func</strong> <code>set_to_field</code> - This method gets the value from the dataset cell</li>
    <li><strong>func</strong> <code>get_from_field</code> - This method gets the value from the dataset cell</li>
    <li><strong>func</strong> <code>add_row</code> - This method adds a new row to the dataset</li>
    <li><strong>func</strong> <code>get_row</code> - This method returns a row of the dataset in dictionary format, where the keys are the column names and the values are the values in the columns</li>
    <li><strong>func</strong> <code>delete_row</code> - This method delete row from dataset</li>
    <li><strong>func</strong> <code>Column</code> - This method summarizes the values from the columns of the dataset and returns them as a list of tuples</li>
    <li><strong>func</strong> <code>add_column</code> - This method adds the column to the dataset on the right</li>
    <li><strong>func</strong> <code>get_column</code> - This method summarizes the values from the columns of the dataset and returns them as a list of tuples</li>
    <li><strong>func</strong> <code>rename_column</code> - This method renames the column in the dataset</li>
    <li><strong>func</strong> <code>delete_column</code> - This method removes the column from the dataset</li>
    <li><strong>func</strong> <code>set_columns_types</code> - This method converts column types</li>
    <li><strong>func</strong> <code>set_column_type</code> - This method converts column type</li>
    <li><strong>func</strong> <code>get_column_stat</code> - This method returns statistical analytics for a given column</li>
    <li><strong>func</strong> <code>reverse</code> - This method expands the order of rows in the dataset</li>
    <li><strong>func</strong> <code>fillna</code> - This method automatically fills in "null" values: for "int" -> 0, for "float" -> 0.0, for "str" -> "-".</li>
    <li><strong>func</strong> <code>equals</code> - </li>
    <li><strong>func</strong> <code>diff</code> - </li>
    <li><strong>func</strong> <code>split</code> - This method automatically divides the DataSet into a list of DataSets with a maximum of "count" rows in each </li>
    <li><strong>func</strong> <code>sort_by_column</code> - This method sorts the dataset by column "column_name" </li>
    <li><strong>func</strong> <code>get_correlations</code> - This method calculate correlations between columns</li>
    <li><strong>func</strong> <code>get_DataFrame</code> - This method return dataset as pd.DataFrame</li>
    <li><strong>func</strong> <code>join_DataFrame</code> - This method attaches a new dataset to the current one (at right)</li>
    <li><strong>func</strong> <code>concat_DataFrame</code> - This method attaches a new dataset to the current one (at bottom)</li>
    <li><strong>func</strong> <code>concat_DataSet</code> - This method attaches a new dataset to the current one (at bottom)</li>
    <li><strong>func</strong> <code>update_dataset_info</code> - This method updates, the analitic-statistics data about already precalculated columns</li>
    <li><strong>func</strong> <code>create_empty_dataset</code> - This method creates an empty dataset</li>
    <li><strong>func</strong> <code>create_dataset_from_list</code> - This method creates a dataset from list of columns values</li>
    <li><strong>func</strong> <code>load_DataFrame</code> - This method loads the dataset into the DataSet class</li>
    <li><strong>func</strong> <code>load_csv_dataset</code> - This method loads the dataset into the DataSet class</li>
    <li><strong>func</strong> <code>load_excel_dataset</code> - This method loads the dataset into the DataSet class</li>
    <li><strong>func</strong> <code>load_dataset_project</code> - This method loads the dataset into the DataSet class</li>
    <li><strong>func</strong> <code>export</code> - This method exports the dataset as DataSet Project</li>
    <li><strong>func</strong> <code>to_csv</code> - This method saves pd.DataFrame to .csv file</li>
    <li><strong>func</strong> <code>to_excel</code> - This method saves pd.DataFrame to excel file</li>
    <li><strong>func</strong> <code>__get_column_type</code> - This method learns the column type</li>
    <li><strong>func</strong> <code>__read_dataset_info_from_json</code> - This method reads config and statistics info from .json file</li>
    <li><strong>func</strong> <code>__update_dataset_base_info</code> - This method updates the basic information about the dataset
    <li><strong>static</strong> <code>__dif_lists_index</code> - </li>
    <li><strong>static</strong> <code>__read_from_csv</code> - </li>
    <li><strong>static</strong> <code>__read_from_xlsx</code> - </li>
    <li><strong>static</strong> <code>get_excel_sheet_names</code> -</li>
    <li><strong>func</strong> <code>merge_two_dicts</code> - This method merge two dicts</li>
</ul>
</details>

<details>
  <summary>class ColumnNum</summary>
  <ul>
    <li><strong>func</strong> <code>__init__</code> - This method init a class work</li>
    <li><strong>func</strong> <code>__add__</code> - This method adds a value to a number (+) [For each cell in column]</li>
    <li><strong>func</strong> <code>__sub__</code> - This method subtracts the value from the number (-) [For each cell in column]</li>
    <li><strong>func</strong> <code>__mul__</code> - This method multiplies the value by a number (*) [For each cell in column]</li>
    <li><strong>func</strong> <code>__floordiv__</code> - This method divides the value by a number (//)  [For each cell in column]</li>
    <li><strong>func</strong> <code>__div__</code> - This method divides the value by a number (/) [For each cell in column]</li>
    <li><strong>func</strong> <code>__mod__</code> - This method gets the remainder from dividing the value by a number (%) [For each cell in column]</li>
    <li><strong>func</strong> <code>__pow__</code> - This method is to raise values to the power of a number (**) [For each cell in column]</li>
    <li><strong>func</strong> <code>__round__</code> - This method rounds the value to the specified precision [For each cell in column]</li>
    <li><strong>func</strong> <code>__floor__</code> - This method rounds the value to the nearest smaller integer [For each cell in column]</li>
    <li><strong>func</strong> <code>__ceil__</code> - This method rounds the value to the nearest bigger integer [For each cell in column]</li>
    <li><strong>func</strong> <code>__trunc__</code> - This method truncates the value to an integer [For each cell in column]</li>
    <li><strong>func</strong> <code>__instancecheck__</code> - This method checks is instance type is DataSet</li>
    <li><strong>func</strong> <code>__len__</code> - This method returns count of elements in column</li>
    <li><strong>property</strong> <code>type</code> - This property returns a type of column</li>
    <li><strong>func</strong> <code>values</code> - This method returns column values as a list</li>
    <li><strong>func</strong> <code>add</code> - This method adds a value to a number (+) [For each cell in column]</li>
    <li><strong>func</strong> <code>sub</code> - This method subtracts the value from the number (-) [For each cell in column]</li>
    <li><strong>func</strong> <code>mul</code> - This method multiplies the value by a number (*) [For each cell in column]</li>
    <li><strong>func</strong> <code>floordiv</code> - This method divides the value by a number (//) [For each cell in column]</li>
    <li><strong>func</strong> <code>div</code> - This method divides the value by a number (/) [For each cell in column]</li>
    <li><strong>func</strong> <code>mod</code> - This method gets the remainder from dividing the value by a number (%) [For each cell in column]</li>
    <li><strong>func</strong> <code>pow</code> - This method is to raise others to the value of a number (**) [For each cell in column]</li>
    <li><strong>func</strong> <code>round</code> - This method rounds the value to the specified precision [For each cell in column]</li>
    <li><strong>func</strong> <code>floor</code> - This method rounds the value to the nearest smaller integer [For each cell in column]</li>
    <li><strong>func</strong> <code>ceil</code> - This method rounds the value to the nearest bigger integer [For each cell in column]</li>
    <li><strong>func</strong> <code>trunc</code> - This method truncates the value to an integer [For each cell in column]</li>
</ul>
</details>


<details>
  <summary>class ColumnNumStat</summary>
  <ul>
    <li><strong>func</strong> <code>__init__</code> - 
    <li><strong>func</strong> <code>__str__</code> - 
    <li><strong>func</strong> <code>__len__</code> - This method returns the len[count] of values in this column
    <li><strong>property</strong> <code>column_name</code> - This method return the name of current column 
    <li><strong>property</strong> <code>type</code> - This method returns type of column 
    <li><strong>property</strong> <code>dtype</code> - This method returns the real type of column 
    <li><strong>property</strong> <code>count</code> - This method returns count of values in this column 
    <li><strong>property</strong> <code>unique_count</code> - This method returns count of unique values in this column 
    <li><strong>property</strong> <code>nan_count</code> - This method returns count of NaN values in this column 
    <li><strong>property</strong> <code>is_extended</code> - 
    <li><strong>func</strong> <code>get_num_stat</code> - 
    <li><strong>func</strong> <code>min</code> - This method return minimal value of column
    <li><strong>func</strong> <code>max</code> - This method return maximal value of column
    <li><strong>func</strong> <code>mean</code> - This method return maximal value of column
    <li><strong>func</strong> <code>median</code> - This method return maximal value of column
    <li><strong>func</strong> <code>get_values_distribution</code> - This method returns the percentage of values in the column 
    <li><strong>func</strong> <code>get_math_mode</code> - This method return mathematical mode
    <li><strong>func</strong> <code>get_math_expectation</code> - This method return mathematical expectation
    <li><strong>func</strong> <code>get_math_dispersion</code> - This method return mathematical dispersion
    <li><strong>func</strong> <code>get_math_sigma</code> - get_math_sigma
    <li><strong>func</strong> <code>get_coef_of_variation</code> - get_coef_of_variation
    <li><strong>func</strong> <code>get_Z_score</code> - This method return mathematical sigma
    <li><strong>func</strong> <code>__get_column_type</code> - This method learns the column type
    <li><strong>staticmethod</strong> <code>__get_nan_count</code> - This method calculate count of NaN values
</ul>
</details>
