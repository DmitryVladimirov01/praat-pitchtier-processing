import os
from tqdm import tqdm
import numpy as np
from time import sleep
from make_lists import make_lists
from sklearn.model_selection import train_test_split
import config


path = config.PATH

def make_pairs_v5(vectors, labels):
    """
    Данная функция делает пары значений для выборки
    :param vectors:
    :param labels:
    :return:
    """
    pair_vectors = []
    pair_labels = []

    # задаются метки разных классов
    classes = len(np.unique(labels))
    # создается список индексов для каждой отдельной метки
    print(labels)
    idx = np.array([np.where(labels == i+1)[0] for i in range(0, classes)])

    for attempt in range(4):
        try:
            for current_idx in tqdm(range(len(vectors))):
                # выбирается одно аудио-значение и его метка
                current_vector = vectors[current_idx]
                current_label = labels[current_idx] - 1
                # случайно выбирается вектор значений этого же класса
                positive_idx = np.random.choice(idx[current_label])
                positive_vector = vectors[positive_idx]
                # добавляем одинаковую пару в список и добавлем метку для пары
                pair_vectors.append([current_vector, positive_vector])
                pair_labels.append([1])
                # отбираем индексы из другого класса и случайно выбираем оттуда вектор
                negative_idx = np.where(labels != current_label)[0]
                negative_vector = vectors[np.random.choice(negative_idx)]
                # добавляем неодинаковую пару и ее метку
                pair_vectors.append([current_vector, negative_vector])
                pair_labels.append([0])
                str_error = None
            print("Пары созданы успешно")
            break

        except IndexError:
            str_error = IndexError
            print(f"[{attempt+1}] Создать пары не получилось, программа попробует снова...")
        if str_error:
            sleep(2)
        else:
            break
    return (np.array(pair_vectors), np.array(pair_labels))

if __name__ == '__main__':
    vectors, labels = make_lists(path)
    trainX, textX, trainY, testY = train_test_split(vectors, labels)
    train_pairs = make_pairs_v5(trainX, trainY)
    print(train_pairs)

    test_pairs = make_pairs_v5(textX, testY)
    print(test_pairs)