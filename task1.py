import time
from multiprocessing import freeze_support

import pandas as pd

from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
manager.create_DataSet("table")
manager.DataSet("table").add_row(new_row={"A": "   abc abc    "})
manager.DataSet("table").add_row(new_row={"A": np.nan})
manager.DataSet("table").add_row(new_row={"A": "345"})
manager.DataSet("table").add_row(new_row={"A": "456"})
manager.DataSet("table").add_row(new_row={"A": "567"})
manager.DataSet("table").add_row(new_row={"A": "678"})

manager.DataSet("table").get_column(column_name="A").upper().lower().
for row in manager.DataSet(dataset_name="table"):
    print(row)



