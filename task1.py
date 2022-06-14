import time
import pandas as pd
from scipy import stats
import RA.DataSet.ColumnStr
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")

manager.create_DataSet(dataset_name="test")
manager.DataSet('test').add_row(new_row={"a": 5, "b": 10})
manager.DataSet('test').add_row(new_row={"a": 15, "b": 110})

manager.create_DataSet(dataset_name="test1")
manager.DataSet('test1').add_row(new_row={"a": 5, "c": 10})
manager.DataSet('test1').add_row(new_row={"a": 15, "c": 110})

print(manager.DataSet('test').equals(dataset=manager.DataSet('test1')))
quit()
print(manager.DataSet('test').get_row(index=1))
print(manager.DataSet('test').get_from_field(column="a", index=1))
# manager.DataSet('test').delete_row(index=0)
print(manager.DataSet('test').head())
print(len(manager.DataSet('test')), manager.DataSet('test').columns_count)


