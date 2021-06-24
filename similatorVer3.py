'''
Данная программа позволяет сравнивать два голоса
Она получает на вход файлы csv, полученные при помощи скрипта для Praat
скрипт: get_from_directory_pitсh_to_csv.praat
И требует предварительно запуска этого скрипта
После запуска этого скрипта необходимо указать заменить папку в переменной PATH на ту,
в которую были помещены файлы .csv

Затем программа начнет обработку этих файлов,
Обратите внимание, что в ходе работы программы может возникнуть ошибка разрешения
Переместите тогда вашу папку с файлами .csv в подпапку проекта
Данная ошибка возникает из-за особенности работы Windows

Так как простое косинусное сходство не дает желаемых результатов

Возможные улучшения:
- обеспечить внутреннюю поддержку Praat через интерфейс python
- применение другого способа паддинга ( растягивание длины вектора)

Данная программа писалась и тестировалась на Windows 10
И так как на Windows принципиально отличается архитектура директорий от OS семейства *NIX,
то убедительная просьба работать именно на данной операционной системе

Основной прадигмой было функциональное программирование, поэтому в данной программе практически отсутствуют классы

Данная программа является годовым проектом
и была разработана исключительно в образовательных целях
'''
import glob
from tqdm import tqdm
import re
import os
import csv
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from pandas.errors import EmptyDataError
from scipy.spatial.distance import cosine
from collections import OrderedDict

#  Здесь содержатся две папки, в которых находятся данные,
# одна содержит тестовые данные, на которых программа и писалась
# вторая содержит основное количество файлов
#test
PATH = r"C:\Users\Дмитрий\PycharmProjects\NeuralVerifier\test\csv\*.csv"
# main
# PATH = r"C:\Users\Дмитрий\PycharmProjects\NeuralVerifier\ALL_DATA\csv\*.csv"


def file_preprocessing(path):
    """
    Выполняет свою работу только один раз!
    Данная функция делает предобработку файлов
    Он переписывает файлы, полученные в праате,
    причем она переписывает их только в том случае,
    если обнаруживает, что они начинаются с артефакта в виде ooTextFile
    удаляет первые 3 строчки с ненужной информацией
    убирает первый столбец со временем,
    так как данная информация не нужна,
    длительность в секундах не так принципиальна
    оставляет только значения f0
    :param path: путь к файлам
    :return: ничего
    """
    print("Начало предобработки")
    files = glob.iglob(path)
    try:
        for file in tqdm(files):
            with open(file, "r") as f:
                lines = f.read().split('\n')
                if lines[0] == '"ooTextFile"':
                    lines = lines[3:]
                    with open(file, "w") as fout:
                        for line in lines:
                            new_line = re.sub("\d+\.\d+\t", "", line)
                            fout.write(new_line + '\n')
    except PermissionError:
        print(f"Пожалуйста, переместите файлы в подпапку проекта, например, {os.getcwd()}\csv")
        exit()
    print("\nПредобработка файлов завершена\n")


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
    files = glob.iglob(path)
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
    # Символом паддинга я выбрал 1, так как по смыслу необходимо будет выполнять умножение, и это отличный способ регуляризации
    df = df.fillna(value=1)
    # нумерация начинается с нуля
    print(f"Максимальное количество временных промежутков: {len(df.index)}")
    dictionary = df.to_dict(orient='list')
    print("Конец записи в словарь")
    return dictionary


def make_pairs(dictionary):
    """
    Функция создает список всех пар объектов
    Разбитие на пары выбрано исходя из задачи и архитектуры программы
    :param dictionary: глобальный словарь с значениями
    :return: список с кортежами пар данных из словаря
    """
    pairValues = []
    pairLabels = []
    labels = dictionary.keys()
    values = dictionary.values()
    classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, classes)]
    pairs = list(itertools.combinations(dictionary, 2))
    print(f"Количество разных пар датафреймов: {len(pairs)}")
    print(f"Размер списка пар в байтах: {pairs.__sizeof__()}")
    print(f"То же самое, но в мегабайтах {pairs.__sizeof__()/(1024**2)}")
    print("Все пары созданы")
    return pairs

def main():
    file_preprocessing(PATH)
    dictionary = make_dictionary(PATH)
    pairs = make_pairs(dictionary)
    train_set, test_set = train_test_split(pairs)
    ts_ss_dictionary = make_ts_ss_dictionary(train_set, dictionary)
    sort_dictionary(ts_ss_dictionary)
    check_list = transform_to_yes_no_list(pairs)
    print(check_list)


# выбрасывает ошибку при прохождении основного датасета
# RuntimeWarning: invalid value encountered in double_scalars
#   dist = 1.0 - uv / np.sqrt(uu * vv)
# непонятно, где именно тратится все время
def make_ts_ss_dictionary(pairs, dictionary):
    """
    Функция создает новый словарь, в который помещает
    пару дикторов и меру их сходства

    Изначально в качестве меры сходства был выбран нестандартный метод TS_SS,
    однако из-за его использования архитектура программы значительно усложнилась и была потеряна
    адекватная читаемость результатов программы

    Подробнее, почему был использован метод TS_SS читайте в документации
    :param pairs: кортеж пар словарей
    :return: словарь с мерой сходства, где ключом выступает пара словарей
    """
    def similarity(tuple):
        """
        Вспомогательная функция подсчета косинусной меры
        Написана для облегчения чтения кода
        Используется только внутри фунцкции make_ts_ss_dictionary()
        :param tuple: кортеж из пары словарей
        :return: число, показывающее косинусное сходство
        """
        array1, array2 = np.array(dictionary[tuple[0]]), np.array(dictionary[tuple[1]])
        sim = 1 - cosine(array1, array2)
        return sim

    print("Начало вычисления сходства")
    ts_ss_dictionary = {}
    for pair in tqdm(pairs):
        ts_ss_dictionary[pair] = similarity(pair)
    print("Конец вычисления сходства")
    return ts_ss_dictionary


def sort_dictionary(ts_ss_dictionary):
    """

    :param ts_ss_dictionary:
    :return:
    """
    def print_sorted(sort_dict):
        """
        Вспомогательная функция печати отсортированного словаря
        Выведена в отдельную функцию для облегчения включения и выключения данной опции
        :param sort_dict:
        :return:
        """
        count = 0
        for item in tqdm(sort_dict):
            user1 = re.search("user\d+", item[0][0]).group()
            user2 = re.search("user\d+", item[0][1]).group()
            print(user1, user2, item[1])
            if user1 == user2:
                count += 1
                print("YES")
            else:
                print("NO")
        print(f"{count} / {len(sort_dict)}")

    sort_dict = sorted(ts_ss_dictionary.items(), key=lambda x: x[1], reverse=True)
    print_sorted(sort_dict)
    return sort_dict


def transform_to_yes_no_list(pairs):
    check_list = np.array([])
    for pair in tqdm(pairs):
        user1 = re.search("user\d+", pair[0]).group()
        user2 = re.search("user\d+", pair[1]).group()
        if user1 == user2:
            check_list = np.append(check_list, [1])
        else:
            check_list = np.append(check_list, [0])
    return check_list


def transform_to_global_list(pairs):
    global_list_pairs = np.array([])


class NeuralNetwork:
    def __init__(self):
        pass
    def train(self):
        pass
    def query(self):
        pass


if __name__ == '__main__':
    main()