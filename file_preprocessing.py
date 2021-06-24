from glob import iglob
from tqdm import tqdm
import re
import os


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
    files = iglob(path)
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


if __name__ == '__main__':
    print("Данный скрипт не предназначен для отдельного использования")