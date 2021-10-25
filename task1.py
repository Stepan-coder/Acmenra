import time
from  multiprocessing import freeze_support
from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")
manager = Manager(path=os.getcwd(), project_name="test_project")
manager.create_DataSet(dataset_name="table")
manager.DataSet("table").load_excel_dataset(excel_file="Пример_датасета_для_xакатона_Tender_Hack.xlsx",
                                            sheet_name="exp")
t1 = time.time()
manager.DataSet("table").get_columns_stat(extended=True)
print(time.time() - t1)
print(manager.DataSet("table"))

