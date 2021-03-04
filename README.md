# Нейронный машинный перевод

Цель этого задания обучить систему машинного перевода на основе реккуретных сетей.
Этот кейс также сделан в форме проекта с использованием фреймворка Pytorch Lightning.

## Установка
Чтобы нам было удобнее работать и запускать код, надо первым делом установить пакет в 
режиме разработчика
```bash
pip install -e .
```

`pip install` -- установка пакета в нашу среду.

# Данные

Взят корпус параллельных текстов OpenSubtitles. Процесс загрузки датасета описан в ```nmt/get_data.py```

# Запуск

## Аргументы скрипта ```cli/run.py```:
- directory — папка с данными, включая токенизаторы  
- checkpoint_path — путь для сохранения чекпоинта модели 
- state_dict_path — путь для сохранения весов модели после обучения  
- project_name — название проекта  
- verbose — дополнительный вывод в консоль некоторых действий  
- load_data — флаг, при указании которого загружаются и подготавливаются данные  
- train_tokenizers — флаг, при указании которого обучаются токенизаторы  
- max_norm — максимальная норма при клиппинге градиентов  
- epochs — количество эпох обучения  
- batch_size — размер батча  
- max_length — максимальная длина последовательности  
- n_batch_accumulate — количество батчей для аккамуляции градиентов  
- seed — число для фиксирования рандомных параметров  
- train_n_pairs — количество пар текстов для обучения  
- valid_n_pairs — количество пар текстов для валидации  
- pad_index — индекс токена PAD  
- bos_index — индекс токена BOS  
- eos_index — индекс токена EOS  
- vocab_size — размер словаря  
- embedding_dim — размерность эмбеддингов  
- model_dim — размерность модели  
- encoder_num_layers — количество слоев RNN в энкодере  
- decoder_num_layers — количество слоев RNN в декодере  
- dropout — вероятность дропаута  
- weight_tying — флаг, при указании которого шарятся веса эмбеддингов целевого языка
и выходной матрицы для предсказания слов
- learning_rate — величина learning rate для оптимизатора  
- weight_decay — величина weight decay для оптимизатора

## Первый запуск
При первом запуске нужно указать флаги ```load_data``` и ```train_tokenizers```, чтобы загрузить данные
и обучить токенизаторы. В последующих запусках эти этапы можно пропустить. В остальные запуски дефолтные значения уже установлены
и можно начать с них.

## Интерактивный режим
Вы можете зайти в интерактивный режим и самим написать тексты на английском языке, чтобы перевести их с помощью обученной вами моделью.
Скрипт находится здесь: ```cli/interact.py```. Вам нужно передать аргумент ```state_dict_path```, в котором нужно указать путь до весов модели, то есть lightning модуля ```LightningSequence2Sequence```.

# Задания

## ```nmt/tokenizer.py```
Необходимо будет подгтовить функцию для преобразования данных в нужный формат для задачи.
Метод ```collate``` должен принимать на вход кортеж, состаящий из двух списков из строк: первый кортеж - 
это тексты на английском языке, второй кортеж - тексты на русском языке. Результатом работы этой функции должны быть три тензора:
```tensor_source_texts_ids``` — тензор, состоящий из индексов слов на английском языке.  
```tensor_target_texts_ids``` — тензор, состоящий из индексов слов на русском языке для подачи этих текстов на вход декодера.
 Тексты включают в себя индекс токена BOS и не включают токен EOS.
```tensor_target_texts_ids_criterion``` — тензор, состоящий из индексов слов на русском языке для подачи этих текстов в расчет функции потерь.
 Тексты включают в себя индекс токена EOS и не включают токен BOS.

### Метод ```source_tokenize```
Будет полезен для использования на этапе инференса. Этот метод должен принимать на вход список текстов на английском языке
 и преобразовывать в тензор, состоящий из индексов слов.

## ```nmt/model```
Реализовать методы: ```forward``` и ```generate```

### ```forward```
Этот метод должен описать всю работу sequence2sequence модели, включая использование энкодера, сохранение его памяти и передачу этой памяти в декодер.
Также работу с самим декодером, то есть обработку ```target``` последовательности, используя память от энкодера.

### ```generate```
Этот метод должен описывать жадную генерацию перевода. На вход принимаются тензор, состоящий из индексов слов на английском языке
и на выход отдается список списков, которые состоят из индексов слов на целевом (русском) языке. Не забудьте в начало генерации добавить в декодер тег BOS, которые говорит о начале генерации.
Индекс тега BOS доступен по ссылке ```self.bos_index```

## ```nmt/lightning```
Нужно задать функцию потерь. Обратите внимание на параметр ```ignore_index```. Нам не нужно считать лосс на падах, поэтому в расчете функции потерь нам стоит их игнорировать.
Также нужно реализовать функцию ```compute_loss```, которая на вход принимает логиты (предсказания нашей модели) и target индексы слов, которые нам нужно предсказать.
Результатом этой функции является расчитанный лосс в формате ```torch.Tensor```.

## ```nmt/metrics```
Нужно реализовать функцию для расчета метрики BLEU. Вам поможет библиотека nltk и функция ```corpus_bleu```.
На вход этой функции подается lightning модель, данные которой вы можете использовать, например, токенизаторы.
Также подается конфиг, который описан выше.

## Запуск тестов
Необходимое, но недостаточное условие сдачи задания -- прохождение всех тестов.
Тесты находятся в папке tests, чтобы их запустить, в корне репозитория введите
```bash
pytest
```

## Сдача задания

Загрузите ваше решение на GitHub и отправьте ссылку на репозиторий и на ваш Wandb проект.
