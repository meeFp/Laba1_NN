# **Создание нейросетевой модели для автоматического выделения и классификации звуковых сигналов**
## Описание
### Цель:
Разработать нейросетевую модель для автоматического выделения и классификации звуковых сигналов.
### О проекте:
Проект по звуковой классификации с использованием нейронных сетей на основе TensorFlow([TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio?hl=ru#setup)) и TensorFlow Hub. В данном проекте рассматривается задача классификации аудиосигналов, таких как собачий и кошачий лай. Проект включает в себя предобработку данных, использование модели YAMNet([YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet?hl=ru)) для извлечения вложений (embeddings), создание и обучение собственной нейронной сети для классификации.

## Используемые библиотеки

- tensorflow
- tensorflow_hub
- tensorflow_io

## Инструкции по установке:
1. Скачайте датасет ESC-50([ESC-50](https://github.com/karolpiczak/ESC-50))
2. Разархивируйте скачанный архив и поместите в гугл диск
3. Установите необходимые библиотеки, выполнив следующие команды:
   ```python
    %pip install tensorflow_io==0.31.0
    %pip install tensorflow==2.11.0
    %pip install tensorflow_hub
    ```

    ```python
    import os

    from IPython import display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_io as tfio
    ```
4. Загрузить модель YAMNet:

   ```python
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
   ```
## Примечательные особенности:
- Используется модель YAMNet для извлечения вложений из аудиофайлов
- Проект предоставляет инструкции по обучению собственной нейронной сети и тестированию собственных аудиофайлов
- Реализована предобработка данных, обучение модели и оценка результатов
-  

