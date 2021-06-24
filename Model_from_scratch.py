import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import *
from make_lists import make_lists
from make_pairs_v6 import *
import  config

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

path = config.PATH

# vectors, labels = make_lists(path)
# pairs = make_pairs_v6(vectors, labels)
# train, test = train_test_split(pairs)
#
# train_shape = train.shape
# test_shape = test.shape
# print(train_shape, test_shape)

# model = Model()


vectors, labels = make_lists(path)
pairs = make_pairs_v6(vectors, labels)
pair_vectors, pair_labels = split_pairs_to_vectors_and_labels(pairs)
# print(pair_vectors)


print(pair_vectors.shape)
print(len(glob(path)))
# print(tensor(pair_vectors))

# должно быть (36, 2, 953)
print(vectors.shape)

shape = (36, 953)

input1 = Input(shape)
input2 = Input(shape)

def build_model(input_shape, output_shape):
    inputs = Input(input_shape)

    layer = Conv1D(activation="relu", kernel_size=1, filters=True)(inputs)
    # layer = MaxPooling1D()(layer)
    layer = Dropout(0.3)(layer)

    layer = Conv1D(activation="relu", kernel_size=1, filters=True)(layer)
    # layer = MaxPooling1D()(layer)
    layer = Dropout(0.3)(layer)

    pooled_output = GlobalAvgPool1D()(layer)
    outputs = Dense(output_shape)(pooled_output)

    model = Model(inputs, outputs)
    return model

model = build_model(shape, 1)
plot_model(model)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


vectorA = Input(shape=shape)
vectorB = Input(shape=shape)
featureExtractor = build_model(shape, 1)
featsA = featureExtractor(vectorA)
featsB = featureExtractor(vectorB)
distance = Dot(normalize=True,axes=-1)([featsA, featsB])

