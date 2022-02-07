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
    CAPITAL=0
    LOWER=1
    UPPER=2
    OTHER=3
    
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


def extract_features(val, i, n):
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


# %%
feature_names = list(get_feature_names())
feature_values = [
    [] for _ in feature_names
]
class_values = []

for name_tag in root:
    assert name_tag.tag == 'Name'

    n = len(name_tag)
    for i, name_part_tag in enumerate(name_tag):
        for f_i, value in enumerate(
            extract_features(name_part_tag.text, i, n)
        ):
            feature_values[f_i].append(value)
        class_values.append(name_part_tag.tag)

df = pd.DataFrame()
for i in range(len(feature_names)):
    df[feature_names[i]] = pd.Series(feature_values[i])
df['class'] = pd.Series(class_values, dtype='category')
# df['stem'] = df['stem'].astype('category')

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

classifier = ComplementNB()
scores = cross_val_score(classifier, X, y, scoring='accuracy')

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
    for i, token in enumerate(doc):
        assert isinstance(token, str)
        for f_i, value in enumerate(
            extract_features(token, i, n)
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
        else:
            df_test[feature_names[i]] = pd.Series(feature_values[i])
    return df_test

X_test = build_X_test(
    [
        ['Mauro', 'Camoranezi', 'Jr'],
        ['Mr.', 'Bond,', 'James'],
    ]
)
y_test = classifier.predict(X_test)
y_test

# %% [markdown]
# # Классификация набора токенов 

# %%
