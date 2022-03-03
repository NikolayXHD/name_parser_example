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

# %% id="rZQv8GeKHp45" executionInfo={"status": "ok", "timestamp": 1645301955392, "user_tz": -180, "elapsed": 260, "user": {"displayName": "Nikolay Hidalgo Diaz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi1_siNR5s5IeeMXEevVauMdR6PDCN9Z3UsWxRFQw=s64", "userId": "13222971951237205321"}}
import wikipedia
import random

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

# %% [markdown] id="aIxuw9S7SCa0"
# ## Задача 1

# %% [markdown]
# Так как текст русскоязычный, используем razdel из пакета natasha

# %%
# !pip install natasha

# %%
import razdel

def tokenize(txt):
    assert isinstance(txt, str)
    return razdel.tokenize(txt)


EXAMPLE_TEXT = (
    '- Петя, налей \tCOO3,(OH4)2,\n'
    'просит Кузнецов И.И. - где-то в С.Ш.А. шёл 2022г.?! и т.д. и т.п.'
)

print(
    list(
        token.text for token in tokenize(EXAMPLE_TEXT)
    )
)

# %% [markdown]
# ```
# [+] Числа отдельный токен
#   2022г -> 2022 + г
# [-] химические формулы отдельный токен
#   COO3,(OH4)2 распались на более мелкие токены, скобки и т.д.
# [+] Слова с дефисом - отдельный токен, включая дефис
#   "где-то" попал в 1 токен
# ```
# Требование по химическим формулам сходу не удовлетворено.
# Что будем делать? Метод `razdel.tokenize` не имеет каких-то настроек.
# Документация и исходный код razdel говорят, что алгоритм основан на правилах.
#
# 2 варианта действий. 
# * токенизировать руками
#   + можно написать очень быстро
#   - выигрыш качества за счёт логики razdel будет утерян
# * разобраться в механизме правил razdel, модифицировать список правил так, чтобы хим. формулы были токенизированы как мы хотим
#   + не жертвуем качеством токенизации
#   - может оказаться сложнее чем кажется

# %% [markdown]
# ### Быстрый способ

# %%
import re
import string


def split_words_by_regex(txt):
    return (
        token for token in re.split(r'\s+', txt)
        if not all(char in string.punctuation for char in token)
    )


def count_words_by_regex(txt):
    return sum(1 for _ in split_words_by_regex(txt))

tokens = list(split_words_by_regex(EXAMPLE_TEXT))
print(f'{len(tokens)} слов: {tokens}')

# %% [markdown]
# ```
# [-] Числа отдельный токен
#   "2022г" попал в 1 токен, "г" не был отделён
# [+] химические формулы отдельный токен
#   COO3,(OH4)2 попал в 1 токен
# [+] Слова с дефисом - отдельный токен, включая дефис
#   "что-то" попал в 1 токен
# ```
# Числа можно отделить, например ещё 1 шагом re.split после разделения по пробелам.
# Здесь не делаю в целях экономии времени, т.к. принцип понятен.

# %%
# для русскоязычных текстов string.punctuation вроде бы достаточен,
# но нужно понимать, что даже не особо экзотичный язык, такой
# как испанский, уже потребует более аккуратного подхода:

print(string.punctuation)
print(f'¿: {"¿" in string.punctuation}')

# символ троеточия … может где-то проскользнуть
print(f'…: {"…" in string.punctuation}')

# %%
for topic, article in zip(topics, articles):
    print(f'{topic}: {count_words_by_regex(article)} слов')

# %% [markdown]
# ### Аккуратный способ

# %%
from razdel.segmenters.tokenize import (
    Rule2112,
    TokenSegmenter,
    INT,
    LAT,
    PUNCT,
    RULES,
)
from razdel.rule import Rule, JOIN
from razdel.segmenters.base import DebugSegmenter

class ChemicalRule(Rule):
    """
    Правило соеднинения атомов razdel в токены
    для группировки химических формул в отдельный токен.
    
    <formula> ::= (Latin|Digit|Parenthesis)([,]Latin|Digit|Parenthesis)*
    
    Понятно, что это грубый и, вообще говоря, некорректный способ,
    в частности мы не можем так проконтролировать корректность скобочной структуры
    в формуле.
    """
    name = 'chemical'

    def __call__(self, split):
#         print(
#             f'L3: {split.left_3!r} L2: {split.left_2} L1: {split.left_1}\n'
#             f'D:{split.delimiter!r}\n'
#             f'R1: {split.right_1!r} R2: {split.right_2!r} R3: {split.right_3!r}\n\n'
#         )
        if (
            self._maybe_formula_boundary(split.left_1) 
            and self._maybe_formula_boundary(split.right_1)
        ):
            return JOIN
        
        if (
            self._maybe_formula_punctuation(split.left_1)
            and self._maybe_formula_boundary(split.left_2)
            and self._maybe_formula_boundary(split.right_1)
        ):
            return JOIN
        
        if (
            self._maybe_formula_punctuation(split.right_1)
            and self._maybe_formula_boundary(split.left_1)
            and self._maybe_formula_boundary(split.right_2)
        ):
            return JOIN
        return None
    
    def _maybe_formula_boundary(self, atom):
        if atom is None:
            return False
        return (            
            atom.type in (INT, LAT)
            or atom.text in '()'
        )
    
    def _maybe_formula_punctuation(self, atom):
        if atom is None:
            return False
        return atom.text == ','
    
def create_segmenter():
    segmenter = TokenSegmenter()
    # segmenter.rules.append(rule) нельзя,
    # поле rules ссылается на константу модуля, мы не хотим её менять
    segmenter.rules = [*segmenter.rules, ChemicalRule()]
    return segmenter

def tokenize_patched(txt):
    segmenter = create_segmenter()
    tokens = segmenter(txt)
    return tokens

tokens = tokenize_patched(EXAMPLE_TEXT)
print(list(token.text for token in tokens))

# %% [markdown]
# Цель достигнута, химические формулы считаются одним токеном. Можно рассчитать количество слов.

# %%
import string


def filter_words(tokens):
    return (
        token for token in tokens
        if not all(c in string.punctuation for c in token.text)
    )


def count_words(txt):
    return sum(
        1 for _ in filter_words(tokenize_patched(txt))
    )


print(
    list(
        token.text for token in filter_words(
            tokenize_patched(EXAMPLE_TEXT)
        )
    )
)
print(count_words(EXAMPLE_TEXT))

# %% [markdown]
# ### Овет к задаче 1

# %%
for topic, article in zip(topics, articles):
    print(f'{topic:>24}, razdel: {count_words(article):>3}, regex: {count_words_by_regex(article):>3}')

# %%
# razdel находит не меньше слов, т.к. делит не только по пробелам.
# пример расхождения
initials = 'Иванов П.П.'
print(
    f'{initials} '
    f'razdel: {count_words(initials)} '
    f'regex: {count_words_by_regex(initials)}'
)

# %% [markdown]
# ### 2.1 Удалить стоп-слова
# Будем считать стоп-словами
# - междометия
# - местоимения
# - предлоги
# - союзы
# - частицы
#
# Чтобы определить часть речи, выполним морфологический разбор

# %%
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    Doc,
)

segmenter = Segmenter()  
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def parse_morphology(txt):
    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    return doc

def print_morphology(txt):
    doc = parse_morphology(txt)
    for s in doc.sents:
        s.morph.print()

EXAMPLE_TEXT = 'А мама и папа те самые, и посмотрели бы под раму, как если бы не клюнуло нас кое-что'
print_morphology(EXAMPLE_TEXT)

# %%
STOPWORD_POS = [
    'CCONJ',  # союз
    'SCONJ',  # союз
    'PRON',  # местоимение
    'ADP',  # предлог
    'INTJ',  # междометие
    'AUX',  # частица
    'PART', # частица
    'DET',  # определитель (тот)
    'PUNCT',  # знак препинания
]


def print_stopword_tokens(txt):
    doc = parse_morphology(txt)
    for s in doc.sents:
        for t in s.morph.tokens:
            if t.pos in STOPWORD_POS:
                print(f'{t.pos:>6}: {t.text}')

print_stopword_tokens(EXAMPLE_TEXT)


# %%
def is_not_stopword(token):
    return token.pos not in STOPWORD_POS


def filter_tokens(txt, token_filters):
    doc = parse_morphology(txt)
    return ' '.join(
        t.text
        for s in doc.sents
        for t in s.morph.tokens
        if all(
            token_filter(t) 
            for token_filter in token_filters
        )
    )


def remove_stopwords(txt):
    return filter_tokens(txt, [is_not_stopword])
    
print(remove_stopwords(articles[0]))

# %%
articles_stopwords_removed = [
    remove_stopwords(article) for article in articles
]

articles_stopwords_removed

# %% [markdown]
# ### 2.2. Удалить все слова, которые не написаны кириллицей. Можно использовать регулярки.

# %%
import re

CYRILLIC_PATTERN = re.compile(r'[а-я]', re.IGNORECASE)

def is_cyrillic(token):
    """
    Хотя бы 1 буква должна быть кириллицей
    """
    return bool(CYRILLIC_PATTERN.match(token.text))

print(filter_tokens(articles[0], [is_not_stopword, is_cyrillic]))

# %%
acticles_stopwords_removed_cyrillic = [
    filter_tokens(article, [is_not_stopword, is_cyrillic])
    for article in articles
]

acticles_stopwords_removed_cyrillic

# %% [markdown]
# ### 2.3. Лемматизировать.
#   - Задача со звёздочкой - лемматизация с учётом части речи исходного слова. Например, "мыла" -> возможные леммы: "мыло" (если "мыла" - существительное в родительном падеже), "мыть" (если глагол в прошедшем времени и женском роде). Можно (и желательно) использовать готовые библиотеки.

# %%
from natasha import MorphVocab

morph_vocab = MorphVocab()


def get_lemmatized_doc(txt):
    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return doc


def get_lemmatized_txt(txt):
    doc = get_lemmatized_doc(txt)
    return ' '.join(token.lemma for token in doc.tokens)

print_lemmatized_doc('Мама мыла раму, ей не хватало мыла.')

# %%
articles_lemmatized = [
    get_lemmatized_txt(article)
    for article in articles
]

articles_lemmatized

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
