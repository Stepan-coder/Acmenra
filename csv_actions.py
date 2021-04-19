import pandas as pd
import openpyxl

# .csv
def read_from_csv(file_folder, filename):
    return pd.read_csv(file_folder + "\\" + filename, encoding='utf-8', delimiter=";")


def find_in_csv(file_folder, filename, colunm, value):
    df = pd.read_csv(file_folder + "\\" + filename, encoding='utf-8', delimiter=";")
    return [i for i in range(len(df)) if df.at[i, colunm] == value]


def change_in_csv(file_folder, filename, index, colunm, value):
    df = pd.read_csv(file_folder + "\\" + filename, encoding='utf-8', delimiter=";")
    df.loc[index, colunm] = value
    df.to_csv(file_folder + "\\" + filename, index=False, sep=';', encoding='utf-8')


def write_to_csv(file_folder, filename, data, titles):
    df = pd.DataFrame(data, columns=titles)
    df.to_csv(file_folder + "\\" + filename, index=False, sep=';', encoding='utf-8')


def delate_from_csv(file_folder, filename, index):
    df = pd.read_csv(file_folder + "\\" + filename, encoding='utf-8', delimiter=";")
    df = df.drop(index=index)
    df.to_csv(file_folder + "\\" + filename, index=False, sep=';', encoding='utf-8')


def add_to_csv(file_folder, filename, data):
    df = pd.read_csv(file_folder + "\\" + filename, encoding='utf-8', delimiter=";")
    df.loc[len(df)] = data
    df.to_csv(file_folder + "\\" + filename, index=False, sep=';', encoding='utf-8')

#.xlsx
def write_to_exel(file_folder, filename, data, titles):
    df = pd.DataFrame(data, columns=titles)
    df.to_excel(file_folder + "\\" + filename, index=False)