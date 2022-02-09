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

# %% [markdown]
# # Построение обучаемого парсера имён собственных

# %% [markdown]
# ## Загрузка размеченного набора данных

# %%
import urllib.request

urllib.request.urlretrieve(
    (
        "https://raw.githubusercontent.com/"
        "datamade/probablepeople/master/"
        "name_data/labeled/person_labeled.xml"
    ),
    "person_labeled.xml"
)

# %%
import xml.etree.ElementTree as ET

tree = ET.parse('person_labeled.xml')
root = tree.getroot()
root

# %% [markdown]
# ## Решим задачу классификации отдельного токена

# %% [markdown]
# ### Построение набора данных

# %%
import enum

class Casing(enum.Enum):
    CAPITAL='C'
    LOWER='L'
    UPPER='U'
    OTHER='O'
    
    @staticmethod
    def of(val):
        assert isinstance(val, str)
        if len(val) == 0:
            return Casing.OTHER
        if len(val) == 1:
            if val.islower():
                return Casing.LOWER
            if val.isupper():
                return Casing.UPPER
            return Casing.OTHER
        if val[0].isupper():
            if val.isupper():
                return Casing.UPPER
            if val[1:].islower():
                return Casing.CAPITAL
            return Casing.OTHER
        if val.islower():
            return Casing.LOWER
        return Casing.OTHER

for example in ['', 'A', 'a', 'AB', 'Ab', 'aB', 'ab', 'ABC', 'ABc', 'AbC', 'Abc', 'aBC', 'abc']:
    casing = Casing.of(example)
    print(f'\'{example}\': {casing}={casing.value}')

# %%
feature_characters_set = set()

for name_tag in root:
    assert name_tag.tag == 'Name'
    for name_part_tag in name_tag:
        feature_characters_set.update(
            c for c in name_part_tag.text if not c.isalnum()
        )

FEATURE_CHARACTERS = ''.join(sorted(feature_characters_set))
FEATURE_CHARACTERS

# %%
import pandas as pd

def get_feature_names():
    yield 'stem'
    yield 'index_from_start'
    yield 'index_from_end'
    yield 'casing'
    yield 'contains_number'
    
    for c in FEATURE_CHARACTERS:
        yield 'contains_' + c

    # Добавление в набор признаков соседних токенов
    # повышает accuracy распознавания типа отдельных токентов с 70% до 80%
    yield 'prev_casing'
    yield 'prev_contains_number'
    for c in FEATURE_CHARACTERS:
        yield 'prev_contains_' + c

    yield 'next_casing'
    yield 'next_contains_number'
    for c in FEATURE_CHARACTERS:
        yield 'next_contains_' + c


def extract_features(vals, i, n):
    assert isinstance(vals, list)
    val = vals[i]
    assert isinstance(val, str)
    assert isinstance(i, int)
    assert isinstance(n, int)
    
    # stem: 'abc.123_,' -> 'abc'
    stem = ''.join(c for c in val if c.isalnum())
    
    yield stem.lower()
    yield i          # index_from_start
    yield n - 1 - i  # index_from_end
    yield Casing.of(stem).value
    yield any(c for c in stem if c.isdigit())  # contains_number
    for c in FEATURE_CHARACTERS:
        yield c in val
    
    if i > 0:
        prev_val = vals[i - 1]
        prev_stem = ''.join(c for c in prev_val if c.isalnum())
        yield Casing.of(prev_stem).value
        yield any(c for c in prev_stem if c.isdigit())  # contains_number
        for c in FEATURE_CHARACTERS:
            yield c in prev_val
    else:
        yield Casing.OTHER.value
        yield False
        for c in FEATURE_CHARACTERS:
            yield False
    
    if i < n - 1:
        next_val = vals[i + 1]
        next_stem = ''.join(c for c in next_val if c.isalnum())
        yield Casing.of(next_stem).value
        yield any(c for c in next_stem if c.isdigit())  # contains_number
        for c in FEATURE_CHARACTERS:
            yield c in next_val
    else:
        yield Casing.OTHER.value
        yield False
        for c in FEATURE_CHARACTERS:
            yield False


# %%
feature_names = list(get_feature_names())
feature_values = [
    [] for _ in feature_names
]
class_values = []

for name_tag in root:
    assert name_tag.tag == 'Name'
    values = [
        name_part_tag.text
        for name_part_tag in name_tag
    ]
    classes = [
        name_part_tag.tag 
        for name_part_tag in name_tag
    ]
    n = len(values)
    for i in range(n):
        for f_i, value in enumerate(
            extract_features(values, i, n)
        ):
            feature_values[f_i].append(value)
    class_values.extend(classes)

df = pd.DataFrame()
for i in range(len(feature_names)):
    df[feature_names[i]] = pd.Series(feature_values[i])
df['class'] = pd.Series(class_values, dtype='category')

# %% [markdown]
# ### Обзор признаков

# %%
df.info()

# %%
df.head(10)

# %%
df['class'].value_counts()

# %%
df['stem'].value_counts()

# %%
df['index_from_start'].value_counts()

# %%
df['casing'].value_counts()

# %%
df['next_casing'].value_counts()

# %%
df['prev_casing'].value_counts()

# %%
df['contains_"'].value_counts()

# %%
df['next_contains_"'].value_counts()

# %%
df['prev_contains_"'].value_counts()

# %%
len(df[df['contains_"'] & df['next_contains_"']])

# %%
len(df[df['contains_"'] & df['prev_contains_"']])

# %%
len(df[df['prev_contains_"'] & df['next_contains_"']])

# %% [markdown]
# ### Валидация

# %%
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

X = pd.get_dummies(df.drop('class', axis=1))
y = df['class']

X.head()

# %%
X, y = shuffle(X, y)

classifier_val = ComplementNB()
scores = cross_val_score(classifier_val, X, y, cv=5, scoring='accuracy')

print(scores)

# %% [markdown]
# ### Обучение

# %%
classifier.fit(X, y)

# %% [markdown]
# ### Использование

# %%
stem_values = set(df['stem'].values)
stem_values

def build_X_test(doc):
    assert isinstance(doc, list)

    feature_names = list(get_feature_names())
    feature_values = [[] for _ in feature_names]

    n = len(doc)
    for i in range(n):
        for f_i, value in enumerate(
            extract_features(doc, i, n)
        ):
            feature_values[f_i].append(value)

    df_test = pd.DataFrame().reindex(columns=X.columns)
    for i in range(len(feature_names)):
        if feature_names[i] == 'stem':
            for known_stem_value in stem_values:
                df_test['stem_' + known_stem_value] = pd.Series(
                    [
                        known_stem_value == stem
                        for stem in feature_values[i]
                    ]
                )
        elif feature_names[i] == 'casing':
            for enum_value in Casing:
                df_test['casing_' + enum_value.value] = pd.Series(
                    enum_value.value == casing
                    for casing in feature_values[i]
                )
        elif feature_names[i] == 'next_casing':
            for enum_value in Casing:
                df_test['next_casing_' + enum_value.value] = pd.Series(
                    enum_value.value == casing
                    for casing in feature_values[i]
                )
        elif feature_names[i] == 'prev_casing':
            for enum_value in Casing:
                df_test['prev_casing_' + enum_value.value] = pd.Series(
                    enum_value.value == casing
                    for casing in feature_values[i]
                )
        else:
            df_test[feature_names[i]] = pd.Series(feature_values[i])
    return df_test

X_test = build_X_test(
    ['Mauro', 'Camoranezi', 'Jr'],
)
X_test
# y_test = classifier.predict(X_test)
# y_test

# %% [markdown]
# # Классификация набора токенов 

# %%
list(v for v in Casing)
