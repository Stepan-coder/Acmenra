import time
import pandas as pd

import RA.DataSet.ColumnStr
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
manager.create_DataSet("table")
manager.DataSet("table").add_row(new_row={"A": "123", "B": 123})
manager.DataSet("table").add_row(new_row={"A": "234", "B": 234})
manager.DataSet("table").add_row(new_row={"A": "345", "B": 345})
manager.DataSet("table").add_row(new_row={"A": "456", "B": 456})
manager.DataSet("table").add_row(new_row={"A": "567", "B": 567})
manager.DataSet("table").add_row(new_row={"A": "678", "B": 678})

manager.DataSet("table").get_column("A").add("234")
print(type(manager.DataSet("table").get_column(column_name="A")),
      isinstance(manager.DataSet("table").get_column(column_name="A"), RA.DataSet.ColumnStr.ColumnStr))


for row in manager.DataSet("table"):
    print(row)


