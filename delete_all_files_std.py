from tqdm import tqdm
import numpy as np
import os
from make_dictionary import make_dictionary
import config

path = config.PATH

def __delete_all_files_upper_std(dictionary, __is_delete_function_already_used):
    """

    НЕ ЗАПУСКАТЬ ЭТУ ФУНКЦИЮ ! ! !

    Данная функция предназначена ИСКЛЮЧИТЕЛЬНО для однократного использования во время разработки

    Данная функция удаляет те файлы, которые являются статистическим выбросом

    :param dictionary: словарь с имеенами файлов и содержимым
    :param __is_delete_function_already_used: специальный флаг,
    чтобы данная функция не была случайно использована
    :return: ничего
    """

    def make_length_list(dictionary):
        length_list = []
        for item in tqdm(dictionary.items()):
            counter = 0
            for i in item[1]:
                if i != 1:
                    counter += 1
            length_list.append(counter)
        length_list = np.array(length_list)
        return length_list

    # значение должно быть передано напрямую
    if not __is_delete_function_already_used:
        ask = input("Вы собираетесь использовать особую функцию очистки.\n Она может существенно исказить ваши данные.\n Надеюсь, вы это понимаете.\nЕсли вы хотите продолжить, введите YES.\nЕсли нет, просто нажмите ENTER:")
        if ask == "YES":
            length_list = make_length_list(dictionary)
            std = np.std(length_list)
            mean = np.mean(length_list)
            k = 3
            print(f"среднее: {mean}, стандартное отклонение: {std}")

            counter = 0

            current_dir = os.getcwd()
            for item in dictionary.items():
                for i, elem in enumerate(item[1]):
                    if i > k * std + mean and elem != 1:
                        counter += 1
                        os.remove(current_dir + "\\ALL_DATA\\csv\\" + item[0])
                        print(f"Файл {item[0]} удален")
                        break
                    elif i > k * std + mean and elem == 1:
                        break
            print(f"Удалено {counter} файлов")
            dictionary = make_dictionary(path)
            return dictionary
        else:
            return dictionary
    else:
        print("Вы не можете использовать эту функцию, измените __is_delete_function_already_used")
        return dictionary