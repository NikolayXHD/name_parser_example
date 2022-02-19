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

# %% [markdown] id="9iMsVLP53p85"
# ## Привет!
#
# В этой тетрадке - упрощенные примеры задач, которые будут полезны при решении ежедневных задач: обработка и анализ текстов, построение разных моделей машинного обучения или нейросетей для классификациии документов по тематикам, поиска похожих текстов, кластеризации текстов, оценки качества перевода, и еще много чего другого.
#
# Интересно посмотреть твой подход к их решению.
#
# _**Полезно знать:** предобработка текстов, токенизация, лемматизация, морфологическая разметка, word2vec, fasttext, разметка именованных сущностей (NER), классификация текстов, языковые модели / модели эмбедингов, классический ML, Deep Learning: RNN, CNN, Transformers (BERT и иже с ним), предобученные модели (например, из Huggingface transformers) и файнтюнинг под конкретные задачи и др._
#
# _____

# %% [markdown] id="82oIWlya6A8K"
# ### Часть 1. Общий анализ текста и предобработка
#
# **NB!** Можно использовать всё, что угодно - от своих методов до готовых моделей и библиотек.

# %% colab={"base_uri": "https://localhost:8080/"} id="JuujjvjRDexJ" executionInfo={"status": "ok", "timestamp": 1645301941075, "user_tz": -180, "elapsed": 6877, "user": {"displayName": "Nikolay Hidalgo Diaz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi1_siNR5s5IeeMXEevVauMdR6PDCN9Z3UsWxRFQw=s64", "userId": "13222971951237205321"}} outputId="f402d9f2-55b3-461c-946b-6198c6402ad8"
# !pip install wikipedia

# %% [markdown] id="AkbVn-dQHGxS"
# Возьмём несколько текстов из Википедии.
#
# Для удобства - первые `max(N, 10)` предложений после заголовка.

# %% id="nxItTfgYH8tf" executionInfo={"status": "ok", "timestamp": 1645301949622, "user_tz": -180, "elapsed": 646, "user": {"displayName": "Nikolay Hidalgo Diaz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi1_siNR5s5IeeMXEevVauMdR6PDCN9Z3UsWxRFQw=s64", "userId": "13222971951237205321"}}
import wikipedia
import random

# %% id="rZQv8GeKHp45" executionInfo={"status": "ok", "timestamp": 1645301955392, "user_tz": -180, "elapsed": 260, "user": {"displayName": "Nikolay Hidalgo Diaz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi1_siNR5s5IeeMXEevVauMdR6PDCN9Z3UsWxRFQw=s64", "userId": "13222971951237205321"}}
import wikipedia

def get_wiki_articles(articles, lang='ru', n=10):
    """Extract text from a Wikipedia page.

    Parameters
    ----------
    articles : list
        A ist of Wikipedia artices to parse.
    """
    assert isinstance(articles, list), \
           f"`articles` should be a list of titles, not {type(articles)}."
    wikipedia.set_lang(lang)

    summaries = []

    for x in articles:
        try:
            summary = wikipedia.summary(x, sentences=n)
        except wikipedia.DisambiguationError as e:
            # If DisambiguationError occurs - just take any page
            p = random.choice(e.options)
            summary = wikipedia.page(p, sentences=n)
        summaries.append(summary)
        
    return summaries


# %% [markdown] id="oza58NW8IgsI"
# Возьмем несколько абсолютно случайных тематик.
#

# %% id="SPKv_q2tC5qG"
topics = [
          'Импрессионизм',
          'Моне, Клод',
          'Франция',
          'Россия',
          'Забайкалье',
          'Блины',
          'Оладьи',
          'Фосфор',
          'Апатит',
          'Сосна сибирская кедровая',
          'Корейский кедр'
          ]

# %% id="1vwWjH3tF0rB"
articles = get_wiki_articles(topics)

# %% colab={"base_uri": "https://localhost:8080/"} id="pXW2gA82GAG-" executionInfo={"status": "ok", "timestamp": 1645193176850, "user_tz": -180, "elapsed": 422, "user": {"displayName": "Anastasia Nikiforova", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjE1ctcrIFEnP4IIPPfryFvfXcE0QfgfT90dwj_=s64", "userId": "12129817482230753910"}} outputId="1e50f44d-6084-49ca-cd2f-d8af6639566a"
articles

# %% [markdown] id="u38ybvDOPNZf"
# #### Задачи:
#
# 1. Посчитать, сколько слов в каждом тексте, без учета знаков пунктуации. Числа, химические формулы - отдельный токен. Слова с дефисом - отдельный токен, включая дефис. Т.е. токенизировать по словам, посчитать токены.
#     - *Правильного ответа нет, интересно посмотреть ход решения.*
#
# 1. Для каждого текста:
#     - Удалить стоп-слова.
#     - Удалить все слова, которые не написаны кириллицей. Можно использовать регулярки.
#     - Лемматизировать.
#         - Задача со звёздочкой - лемматизация с учётом части речи исходного слова. Например, `"мыла"` -> возможные леммы: `"мыло"` (если `"мыла"` - существительное в родительном падеже), `"мыть"` (если глагол в прошедшем времени и женском роде). Можно (и желательно) использовать готовые библиотеки.
#
# 1. Найти именованные сущности в текстах (NER). Можно использовать любую готовую библиотеку. Например, Наташу.
#
# 1. Попарно посчитать близость текстов.
#     - Использовать какой-нибудь из знакомых способов векторизации слов или текстов целиком.
#     - Посчитать косинусное расстояние между векторами текстов, чтобы узнать, какие тексты похожи, а какие - нет.
#
# _**Nota bene!** Использовать можно всё, что угодно. Общих решений достаточно, не нужно заморачиваться с частными случаями и излишней предобработкой. Результаты можно выводить любым удобным способом._
#

# %% id="aIxuw9S7SCa0"
### Your code here

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

# %% [markdown] id="BVgKsyssW5CR"
# ### Часть 3. Проверка качества перевода.
#
# Ниже - случайный текст из Википедии в оригинале и его переводы в двух МТ: Google Translate и DeepL.
#
# Предположим, что один из этих переводов - машинный перевод (`mt`), а второй - постредактура переводчиком (`edit`).
#
# На основе этих двух текстов посчитать, как сильно постредактура изменила машинный перевод. Фактически - показать, насколько машинный перевод хороший/плохой.
#
# **NB!** Можно использовать, например, разные метрики для оценки качества перевода, типа BLEU и посложнее.
#
# Сильно заморачиваться не стоит - представь, что нужно asap общее понимание качества МТ.

# %% id="v00yNNr_Cd5f"
# оригинал просто fyi
orig = """Палаццо — итальянский городской дворец-особняк XIII—XVI веков.
Название происходит от Палатинского холма в Риме, где древнеримские императоры возводили свои дворцы.
Палаццо представляет собой тип городского дома-крепости, характерный для итальянского средневековья и эпохи Возрождения.
Маленький дворец называют «палацетто»
Классические ренессансные палаццо XV века можно увидеть во Флоренции.
Такое палаццо представляет собой величественное двух- или трёхэтажное здание сурового вида с мощным карнизом и машикулями.
Маленький дворец называют «палацетто», иногда со сторожевой башней.
Стены оформлены рустом, окна первых этажей забраны решёткой: городская жизнь в то время была неспокойной.
Парадный, приёмный зал для гостей итальянского палаццо именуют форестерией.
"""

# google
mt = """Palazzo - Italian city palace-mansion of the XIII-XVI centuries.
The name comes from the Palatine Hill in Rome, where the ancient Roman emperors built their palaces.
The palazzo is a type of urban fortified house, typical of the Italian Middle Ages and the Renaissance.
The little palace is called "palacetto"
Classic Renaissance palazzos of the 15th century can be seen in Florence.
Such a palazzo is a majestic two- or three-story building of a severe appearance with a powerful cornice and machicolations.
The little palace is called "palacetto", sometimes with a watchtower.
The walls are decorated with rustication, the windows of the first floors are barred: city life at that time was hectic.
The front, reception hall for guests of the Italian palazzo is called the forestry.
"""

# deepl
edited = """Palazzo is an Italian city palace from the 13th to 16th centuries.
The name comes from the Palatine Hill in Rome, where Roman emperors built their palaces.
The palazzo is a type of urban fortress house characteristic of the Italian Middle Ages and Renaissance.
A small palazzo is called a "palazzetto."
Classic Renaissance palazzo of the 15th century can be seen in Florence.
Such a palazzo is a stately two- or three-story building of a severe appearance with a powerful cornice and machicolations.
The small palazzo is called "palazzetto", sometimes with a watchtower.
The walls are decorated with rustication, the windows of the first floors are barred: urban life at that time was turbulent.
The front, reception hall for guests of the Italian palazzo is called the foresteria.
"""

# %% id="K1pRI4ukbSCW"
### Your code here
