"""
Данная часть программы была импортирована из https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
Была убрана часть про третью часть данных и преобразована к двум объектам
"""

from numpy import dstack
from pandas import read_csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt
# from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.layers.merge import Dot
import keras.backend as K
from sklearn.model_selection import train_test_split
from Ver4 import *
from make_pairs_v6 import *

vectors, labels = make_lists(path)

pairs = make_pairs_v6(vectors, labels)
train, test = train_test_split(pairs)
trainX, trainY = split_pairs_to_vectors_and_labels(train)
testX, testY = split_pairs_to_vectors_and_labels(test)


epochs, batch_size = 10, 32
shapeTrainX = trainX.shape
shapeTrainY = trainY.shape

n_timesteps, n_features = trainX.shape[0], trainX.shape[1]
# первый вход
inputs1 = Input(shape=(n_timesteps,n_features))
conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)

# второй вход
inputs2 = Input(shape=(n_timesteps,n_features))
conv2 = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)

# соединение
distance = Dot(axes=-1, normalize=True)([flat1, flat2])
outputs = Dense(1)(distance)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

print(model.summary())
layers = model.layers
print(layers)

# сохранение графика
plot_model(model, show_shapes=True, to_file='siamese.png')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# pair_list = [pairTrain[0], pairTrain[1], labelTrain]

# обучение модели
history = model.fit([trainX[0], trainX[1]], trainY, epochs=epochs, batch_size=batch_size)
# оценка модели
_, accuracy = model.evaluate([testX[0], testX[1]], testY, batch_size=batch_size)
plot_training(history, config.PLOT_PATH)
model.save(config.MODEL_PATH)


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)





