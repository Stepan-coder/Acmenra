import os

import numpy as np

from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


manager = Manager(path=os.getcwd(), project_name="Test_Dataset")
manager.add_DataSet(dataset=DataSet(dataset_name="test", show=True))
manager.DataSet("test").create_dataset_from_list(data=[["1", "1"],
                                                       ["2", "2"],
                                                       ["3", "3"]],
                                                 columns=["A", "B"])
manager.DataSet("test").head()
# print()
# # manager.DataSet("test").delete_row(index=0)
# manager.DataSet("test").fillna()
# manager.DataSet("test").tail(10)