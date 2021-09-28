import os

import numpy as np

from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


manager = Manager(path=os.getcwd(), project_name="Test_Dataset")
manager.add_DataSet(dataset=DataSet(dataset_name="test", show=True))

manager.DataSet(dataset_name="test").create_empty_dataset()
# manager.DataSet("test").add_column(column="A", values=["1", "2", "3"])
manager.DataSet("test").add_column(column="B", values=[1, 2, 3])
# manager.DataSet("test").get_column_info(column_name="A", extended=True)

manager.add_DataSet(dataset=DataSet(dataset_name="test2", show=True))
manager.DataSet(dataset_name="test2").create_empty_dataset()
# manager.DataSet("test2").add_column(column="A", values=["3", "2", "1"])
manager.DataSet("test2").add_column(column="B", values=[3, np.nan, 1])
# manager.DataSet("test2").get_column_info(column_name="B", extended=True)

manager.DataSet("test").concat_DataSet(dataset=manager.DataSet("test2"))
# manager.DataSet("test").add_row(new_row={"some1": 5, "some12": 5})
# manager.DataSet("test").delete_row(index=1)
# manager.DataSet("test").head(10)
print()
# manager.DataSet("test").delete_row(index=0)
manager.DataSet("test").fillna()
manager.DataSet("test").tail(10)