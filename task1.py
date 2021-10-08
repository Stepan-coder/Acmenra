from RA.Manager.Manager import *
from RA.DataSet.DataSet import *


warnings.filterwarnings("ignore")

manager = Manager(path=os.getcwd(), project_name="test_project")
manager.create_DataSet(dataset_name="Original DataSet")
manager.create_DataSet(dataset_name="Original DataSet1")
manager.DataSet("Original DataSet").add_row(new_row={"A": False, "B": True})
manager.DataSet("Original DataSet").add_row(new_row={"A": False, "B": True})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 3, "B": 3})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 4, "B": 1})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 5, "B": 2})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 6, "B": 3})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 7, "B": 1})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 8, "B": 2})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 9, "B": 3})
# manager.DataSet("Original DataSet").add_row(new_row={"A": 10, "B": 3})

print(manager.DataSet("Original DataSet").get_column_values(column_name="A"))
manager.DataSet("Original DataSet").set_column_types(new_column_types=int)
print(manager.DataSet("Original DataSet").get_column_values(column_name="A"))
manager.DataSet("Original DataSet").set_column_types(new_column_types=bool)
print(manager.DataSet("Original DataSet").get_column_values(column_name="A"))
#
# print(type(manager.DataSet("Original DataSet").get_from_field("A", 3)))
# print(manager.DataSet("Original DataSet"))


# Надо разделить классы колон! Тип пусть у флоатов будут свои методы и свой функционал, у строк - другой, у булев - третий

# Исходя из личного опыта, стало понятно, что задачи всегда разные и что создать платформу с автообучением
# сложно и что в принципе такие уже есть. Печально. НО
# Можно сделать платформу для начинающих датасантистов! Предоставляем услуги, по обработке и визуализыции данных
# Т.е. Чел грузит свой датасет к нам, и с помощью простых кнопок его редактирует (перетаскивает колонки, группирует,
# смотрит что за данные)
# УСКОРЯЕТ РАБОТУ ПО ОЧИСТКЕ И ФИЛЬТРАЦИИ ДАТАСЕТА
# Нужно создать аккаунт на патреоне и пусть туда люди донатят (на развитие проекта), в последствии туда может
# подключиться и бизнес, потому что это удобнее, чем exel
