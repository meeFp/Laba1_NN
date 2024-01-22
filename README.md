# Создание нейросетевой модели для автоматического выделения и классификации звуковых сигналов
## Описание
### Цель:
Разработать нейросетевую модель для автоматического выделения и классификации звуковых сигналов.
### О проекте:
Проект по звуковой классификации с использованием нейронных сетей на основе TensorFlow и TensorFlow Hub. В данном проекте рассматривается задача классификации аудиосигналов, таких как собачий и кошачий лай. Проект включает в себя предобработку данных, использование модели YAMNet для извлечения вложений (embeddings), создание и обучение собственной нейронной сети для классификации.

## Инструкции по установке:

1. Установите необходимые библиотеки, выполнив следующие команды:

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
