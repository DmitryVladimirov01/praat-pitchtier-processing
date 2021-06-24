from itertools import combinations
import numpy as np
from make_lists import make_lists
from sklearn.model_selection import train_test_split
import tensorflow as tf
import config

path = config.PATH


def make_pairs_v6(vectors, labels):
    """
    В данной версии был совершен возврат к itertools,
    так при обработке бага из предыдущей версии возникали новые ошибки

    В данной версии применены списки, как один из быстродействующих способов
    Из 5 версии были импортированы присваивания меток 1 - 0

    """
    items = []
    for item in zip(vectors, labels):
        items.append(item)
    combs = list(combinations(items, 2))
    pairs = []
    for combo in combs:
        # если метки совпадают, то ставится 1
        # print(combo[0][0])
        if combo[0][1] == combo[1][1]:
            pairs.append([combo[0][0], combo[1][0], 1])
        # если нет, то ставится 0
        else:
            pairs.append([combo[0][0], combo[1][0], 0])
    pairs = np.array(pairs)
    return pairs

def split_pairs_to_vectors_and_labels(pairs):
    pair_vectors = np.array([pairs[:, 0], pairs[:, 1]])
    pair_labels = pairs[:, 2]
    return pair_vectors, pair_labels

if __name__ == '__main__':
    vectors, labels = make_lists(path)
    pairs = make_pairs_v6(vectors, labels)
    train, test = train_test_split(pairs)
    # trainX, trainY = split_pairs_to_vectors_and_labels(train)
    # testX, testY = split_pairs_to_vectors_and_labels(test)
    print(pairs)
    
    values1 = pairs[:,0]
    values2 = pairs[:, 1].flatten()
    labels1 = pairs[:, 2]