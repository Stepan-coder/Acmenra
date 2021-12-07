import time
import pandas as pd
from scipy import stats
import RA.DataSet.ColumnStr
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
if manager:
    print(True)
manager.create_DataSet(dataset_name="some_dataset")
# Корреляция работает только с числами... Поэтому, нужно проверять, что вызывается на корреляци.
manager.DataSet("some_dataset").create_dataset_from_list(data=[[1, 2, 3, 4, 5],
                                                               [10, 9, 2.5, 6, 4],
                                                               ["a", "b", "c", "d", "e"],
                                                               reversed([10, 9, 2.5, 6, 4])],
                                                         columns=["a", "b", "c", "d"])
# Обязательно нужно добавить кореляционную матрицу! (Пусть это будет словарь, где ключ это название столбца,
cors = manager.DataSet("some_dataset").get_correlations()
for c in cors:
    print(c, cors)

# manager.DataSet("some_dataset").head()


