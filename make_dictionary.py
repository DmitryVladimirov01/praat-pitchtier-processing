from glob import iglob
from tqdm import tqdm
import csv
import os
import pandas as pd
from pandas.errors import EmptyDataError
import config

path = config.PATH

def make_dictionary(path):
    """
    Функция создает словарь вида {filename}: {values}
    Считывает из файлов .csv все данные и записывает в словарь
    Преобразует его в датафрейм, чтобы сделать паддинг до максимальной длины
    И возвращает формат словаря

    Формат словаря был выбран из-за его скорости
    Датафреймы и словари несоизмеримы по скорости
    А так как объем данных очень большой, то при запуске первой версии программы,
    основанной на Датафреймах, общее время работы должно было составить около 30-35 дней
    :param path: папка, в которой содержатся все основные файлы
    :return: словарь вида {filename}: {values}
    """
    print("Начало записи в словарь")
    files = iglob(path)
    dictionary = {}
    for file in tqdm(files):
        value_list = []
        try:
            with open(file) as f:
                reader = csv.reader(f)
                for row in reader:
                    value_list.extend(row)
            value_list = list(map(float, value_list))
            dictionary[os.path.basename(file)] = value_list
        except EmptyDataError:
            print(f"Ошибка обнаружена в файле:{os.path.basename(file)}")
    df = pd.DataFrame.from_dict(dictionary, orient='index').transpose()
    # Символом паддинга я выбрал 1, так как по смыслу необходимо будет выполнять умножение,
    # к тому же это отличный способ регуляризации
    df = df.fillna(value=1)
    # нумерация начинается с нуля
    print(f"Максимальное количество временных промежутков: {len(df.index)}")
    dictionary = df.to_dict(orient='list')
    print("Конец записи в словарь")
    return dictionary

if __name__ == '__main__':
    dictionary = make_dictionary(path)
    print(dictionary)
