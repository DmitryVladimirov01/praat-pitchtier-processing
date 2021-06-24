"""
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

Возможные ошибки при запуске программы:
- make_pairs может вылетать, это связано с внутренним рандомизатором функции,
в функции не создаются абсолютно все пары, а создается только половина из них,
поэтому при запуске этой функции существует вероятность

Данная программа писалась и тестировалась на Windows 10
И так как на Windows принципиально отличается архитектура директорий от OS семейства *NIX,
то убедительная просьба работать именно на данной операционной системе

Основной прадигмой было функциональное программирование, поэтому в данной программе практически отсутствуют классы

Данная программа является годовым проектом
и была разработана исключительно в образовательных целях
"""

from glob import iglob
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import csv
from pandas.errors import EmptyDataError
import keras
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Lambda
from keras.layers import merge
from keras.layers.merge import concatenate
import keras.backend as K
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

# внутренние фалы проекта
from file_preprocessing import file_preprocessing
from make_dictionary import make_dictionary
# ограниченная поддержка в модуле keras
from make_pairs import make_pairs_v5
from make_pairs_v6 import make_pairs_v6
from make_lists import make_lists
from delete_all_files_std import __delete_all_files_upper_std
# файл с переменными значениями
import config

path = config.PATH

def make_pairs_v4(dictionary):
    def remove_suffix(word):
        return re.search("\d+", word).group()
    labels = list(map(remove_suffix, list(dictionary.keys())))
    print(labels)
    values = dictionary.values()
    # print(values)
    classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, classes)]
    print(idx)





def euclidean_distance(vectors_other):
    '''
    Считает евклидовую дистанцию - аналог обычного косинусного сходства
    реализован на ядре keras, так используется в качестве слоя
    :param vectors_other:
    :return:
    '''
    y_pred, y_true = vectors_other
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)




def evaluate_model(trainX, trainy, testX, testy):
    epochs, batch_size = 10, 32

    (pairTrain, labelTrain) = make_pairs_v5(trainX, trainy)
    (pairTest, labelTest) = make_pairs_v5(testX, testy)

    n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], trainy.shape[0]
    # первый вход
    inputs1 = Input(shape=(n_timesteps,n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # второй вход
    inputs2 = Input(shape=(n_timesteps,n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # соединение
    # distance = merge([flat1, flat2], mode="cos")
    # interpretation
    print(f"FLAT1: {flat1}\n\n\n\n\n")
    distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([flat1, flat2])
    dense1 = Dense(100, activation='relu')(distance)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # сохранение графика
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # обучение модели
    history = model.fit([pairTrain[0], pairTrain[1]], labelTrain, epochs=epochs, batch_size=batch_size)
    # оценка модели
    _, accuracy = model.evaluate([pairTest[0], pairTest[1]], labelTest, batch_size=batch_size)
    model.save(os.getcwd() + "\\model\\siamese_model")
    plot_training(history, os.getcwd()+"\\model\\plot.png")
    return accuracy


def global_plt_show(dictionary):
    data = []
    for value in tqdm(dictionary.values()):
        data.append(value)
    # print(data)
    plt.imshow(data, interpolation='none', aspect="auto")
    plt.show()


def plot_training(H, plot_path):
    """
    Делает историю обучения в виде графика
    :param H:
    :param plot_path: путь к
    :return:
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Точность и ошибка во время тренировки")
    plt.xlabel("Эпоха #")
    plt.ylabel("Ошибка/Точность")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


def main():
    file_preprocessing(path)
    dictionary = make_dictionary(path)
    ###############################
    # служебная часть
    __is_delete_function_already_used = True
    dictionary = __delete_all_files_upper_std(dictionary, __is_delete_function_already_used)
    ################################
    # print(dictionary)

    vectors, labels = make_lists(path)
    pairs = make_pairs_v5(vectors, labels)

    trainX, testX, trainy, testy = train_test_split(vectors, labels)
    # (pairTrain, labelTrain) = make_pairs_v5(trainX, trainy)
    # (paiTest, labelTest) = make_pairs_v5(testX, testy)
    # evaluate_model(trainX, trainy, testX, testy)
    global_plt_show(dictionary)


if __name__ == '__main__':
    main()
