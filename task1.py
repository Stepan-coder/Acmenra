import time
import pandas as pd

import RA.DataSet.ColumnStr
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
if manager:
    print(True)
manager.create_DataSet(dataset_name="some_dataset")
manager.DataSet("some_dataset").create_dataset_from_list(data=[[1, 2, 3],
                                                               [4, 5, 6],
                                                               [7, 8, 9]],
                                                         columns=["a", "b", "c"])

manager.DataSet("some_dataset").head()


