# praat-pitchtier-processing
Данная программа  предназначена для обработки большого количества объектов PitchTier и последующего преобразования для сиамской нейронной сети
Нейронная сеть не получилась из-за некоторых нестыковок в размерностях
Основные функции вынесены в отдельные скрипты для упрощения структуры


Данная работа значительное количество раз подвергалась реинжинирингу и рефакторингу, однако несмотря на это здесь все еще присутствуют части прошлых версий, без которых та или иная часть программы работает неправильно

Ver4.py -- это основной файл программы

config.py -- файл с изменяемыми параметрами, в частности path изменяется отсюда

file_preprocessing.py -- предобработка файлов csv, полученных в Praat

make_dictionary.py  -- (устарел) файл создания словаря данных из .csv файла 

make_lists.py -- аналогичная make_dictionary.py подпрограмма, сделанная на других типах данных

make_pairs.py -- (устарел) пятая версия make_pairs, основана на случайности, но правильно выдает размерности

make_pairs_v6.py -- 6 версия make_pairs, основана на сочетаниях

Model.py -- (неисправен) более рабочая версия модели

Model_from_scratch.py -- (неисправен) альтернативная попытка сделать модель

delete_all_files_std.py -- внутренний модуль удаления ненужных файлов
