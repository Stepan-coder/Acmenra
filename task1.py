import time
import pandas as pd
from scipy import stats
import RA.DataSet.ColumnStr
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
manager.create_DataSet(dataset_name="test")
manager.DataSet("test").create_dataset_from_list(data=[["⁣", "⁣", "⁣"],
                                                       ["⁣", "⁣", "⁣"]],
                                                 columns=["a", "b"])
manager.DataSet("test").export(delimeter=";")
quit()
if manager:
    print(True)
manager.create_DataSet(dataset_name="test")
manager.create_DataSet(dataset_name="test1")
manager.create_DataSet(dataset_name="test2")
# Корреляция работает только с числами... Поэтому, нужно проверять, что вызывается на корреляци.
some = [i for i in range(0, 100000, 1)]
t1 = time.time()
manager.DataSet("test").create_dataset_from_list(data=[some,
                                                       reversed(some)],
                                                 columns=["a", "b"])
manager.DataSet("test1").create_dataset_from_list(data=[reversed(some), some],
                                                  columns=["a", "b"])
print(time.time() -t1)
manager.DataSet("test").concat_DataSet(manager.DataSet("test1"))
# print(manager.DataSet("test").get_DataFrame().head(10))
# # Обязательно нужно добавить кореляционную матрицу! (Пусть это будет словарь, где ключ это название столбца,
# cors = manager.DataSet("test").get_correlations()
# for c in cors:
#     print(c, cors)

# manager.DataSet("some_dataset").head()


