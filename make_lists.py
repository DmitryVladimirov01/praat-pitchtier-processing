from glob import iglob
from tqdm import tqdm
import re
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import config

path = config.PATH

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

def make_lists(path):
    """
    Функция извлечения информации из файлов csv: версия 4.0
    Извлекает информацию в виде двух списков: отдельно сами векторы, отдельно метки
    Предыдущие версии были основаны на датафремах pandas, но они оказались очень медленные,
    а также на словарях. После рефакторинга мною было принято решение перейти на списки
    Версия со словарем пока еще используется в отдельных функциях
    :param path: директория, где лежат файлы
    :return vectors: список векторов значений
    :return labels: список меток дикторов
    """
    glob = iglob(path)
    labels = []
    vectors = []
    # для каждого файла создается метка и извлекаются значения
    for file in tqdm(glob):
        username = re.search("\d+", os.path.basename(file)).group()
        labels.append(int(username))

        df = pd.read_csv(file)
        vector = df.to_numpy().flatten()
        vectors.append(vector)
    # преобразуется в датафрейм для паддинга и регуляризации
    df_val = pd.DataFrame(vectors).fillna(value=1)
    vectors = df_val.to_numpy()

    labels = np.array(labels)

    # return tensor(vectors), tensor(labels)
    return vectors, labels


if __name__ == '__main__':
    vectors, labels = make_lists(path)
    print(vectors)
    print(labels)