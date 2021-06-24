#  Здесь содержатся две папки, в которых находятся данные,
# одна содержит тестовые данные, на которых программа и писалась
# вторая содержит основное количество файлов
import os
# test
# PATH = r"C:\Users\Дмитрий\PycharmProjects\NeuralVerifier\test\csv\*.csv"
# main
PATH = r"C:\Users\Дмитрий\PycharmProjects\NeuralVerifier\ALL_DATA\csv\*.csv"

BASE_OUTPUT = r"C:\Users\Дмитрий\PycharmProjects\NeuralVerifier\model"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])