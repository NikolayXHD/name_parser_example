# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="krTGSvSORYSC"
# _____
# ### Часть 2. Простая задачка на ML
#
# Ниже написан код решения задачи предсказания вероятности выжить на <a href="https://www.kaggle.com/c/titanic">Титанике</a>. Нужно:
#
# 1. Исправить ошибки в решении
#
# 1. Применяя известные способы предобработки данных и алгоритмы машинного обучения из библиотеки `sklearn`, добиться значения целевой метрики (accuracy) > 0.8 на зафиксированной с помощью `random_state` тестовой выборке. 
#
# 1. **Нельзя**: менять `test_size` и `random_state` в `train_test_split`, удалять строки из `train`.
#
# **Описание признаков:**
#
#     PassengerId — идентификатор пассажира
#     Survived — поле, в котором указано, спасся человек (1) или нет (0)
#     Pclass — класс билета (3 - самые дешёвые билеты)
#     Name — имя пассажира
#     Sex — пол пассажира
#     Age — возраст
#     SibSp — содержит информацию о количестве родственников 2-го порядка (муж, жена, братья, сеcтры)
#     Parch — содержит информацию о количестве родственников на борту 1-го порядка (мать, отец, дети)
#     Ticket — номер билета
#     Fare — цена билета
#     Cabin — каюта
#     Embarked — порт посадки (C — Cherbourg, Q — Queenstown, S — Southampton)

# %% colab={"base_uri": "https://localhost:8080/"} id="Q4nBF008PNGn" executionInfo={"status": "ok", "timestamp": 1645195147964, "user_tz": -180, "elapsed": 1438, "user": {"displayName": "Anastasia Nikiforova", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjE1ctcrIFEnP4IIPPfryFvfXcE0QfgfT90dwj_=s64", "userId": "12129817482230753910"}} outputId="be5b3710-81bb-4a99-a593-67502d9bc6e3"
# загрузка обучающей выборки
# !wget --no-verbose https://www.dropbox.com/s/kbdpvcgv58ueks7/train.csv

# %% colab={"base_uri": "https://localhost:8080/"} id="_g91835DPIu-" executionInfo={"status": "ok", "timestamp": 1645195157468, "user_tz": -180, "elapsed": 1692, "user": {"displayName": "Anastasia Nikiforova", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjE1ctcrIFEnP4IIPPfryFvfXcE0QfgfT90dwj_=s64", "userId": "12129817482230753910"}} outputId="b19601c6-b775-40f9-b5f0-af8c0b4fd124"
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
y = train['Survived']

not_number_fields = [col for col in list(train.columns) if train[col].dtype == np.object]

train.drop(columns=not_number_fields, inplace=True)
train.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=5757)

model = LogisticRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)
accuracy_score(y_test, predict)
