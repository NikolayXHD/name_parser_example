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
def get_feature_characters(name_tags):
    feature_characters_set = set()
    for name_tag in name_tags:
        assert name_tag.tag == 'Name'
        for name_part_tag in name_tag:
            feature_characters_set.update(
                c for c in name_part_tag.text if not c.isalnum()
            )
    return ''.join(sorted(feature_characters_set))

FEATURE_CHARACTERS = get_feature_characters(root)
FEATURE_CHARACTERS

# %%
import pandas as pd

def get_feature_names(feature_characters):
    yield 'stem'
    yield 'index_from_start'
    yield 'index_from_end'
    yield 'casing'
    yield 'contains_number'
    
    for c in feature_characters:
        yield 'contains_' + c

    # Добавление в набор признаков соседних токенов
    # повышает accuracy распознавания типа отдельных токентов с 70% до 80%
    yield 'prev_casing'
    yield 'prev_contains_number'
    for c in feature_characters:
        yield 'prev_contains_' + c

    yield 'next_casing'
    yield 'next_contains_number'
    for c in feature_characters:
        yield 'next_contains_' + c


def extract_features(vals, i, n, feature_characters):
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
    for c in feature_characters:
        yield c in val
    
    if i > 0:
        prev_val = vals[i - 1]
        prev_stem = ''.join(c for c in prev_val if c.isalnum())
        yield Casing.of(prev_stem).value
        yield any(c for c in prev_stem if c.isdigit())  # contains_number
        for c in feature_characters:
            yield c in prev_val
    else:
        yield Casing.OTHER.value
        yield False
        for c in feature_characters:
            yield False
    
    if i < n - 1:
        next_val = vals[i + 1]
        next_stem = ''.join(c for c in next_val if c.isalnum())
        yield Casing.of(next_stem).value
        yield any(c for c in next_stem if c.isdigit())  # contains_number
        for c in feature_characters:
            yield c in next_val
    else:
        yield Casing.OTHER.value
        yield False
        for c in feature_characters:
            yield False


# %%
def build_df(name_tags, feature_characters):
    feature_names = list(get_feature_names(feature_characters))
    feature_values = [[] for _ in feature_names]
    class_values = []
    
    for name_tag in name_tags:
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
                extract_features(values, i, n, feature_characters)
            ):
                feature_values[f_i].append(value)
        class_values.extend(classes)

    df = pd.DataFrame()
    for i in range(len(feature_names)):
        df[feature_names[i]] = pd.Series(feature_values[i])
    df['class'] = pd.Series(class_values, dtype='category')
    return df

df = build_df(root, FEATURE_CHARACTERS)

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
# ### Оценим accuracy классификации токенов по отдельности

# %%
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def split_features(df):
    X = pd.get_dummies(df.drop('class', axis=1))
    y = df['class']
    return X, y

X, y = split_features(df)
X.head()


# %%
def create_classifier():
    return ComplementNB()


# %%
X, y = shuffle(X, y)

classifier_val = create_classifier()
scores = cross_val_score(classifier_val, X, y, cv=5, scoring='accuracy')

print(scores)


# %%
class FacadeClassifier:
    """
    Классификатор для имени целиком: list[name_part] -> list[tag]
    Выделен в отдельный класс с состоянием для удобства кросс-валидации
    """
    def fit(self, name_tags):
        """
        Обучиться на множестве xml-тегов
        """
        feature_characters = get_feature_characters(name_tags)
        df = build_df(name_tags, feature_characters)
        classifier = create_classifier()
        X, y = split_features(df)
        classifier.fit(X, y)

        self._feature_characters = feature_characters
        self._df = df
        self._classifier = classifier
        self._stem_values = set(df['stem'])
        self._X = X

    def _build_X_predict(self, names):
        assert isinstance(names, list)
        
        feature_names = list(get_feature_names(self._feature_characters))
        feature_values = [[] for _ in feature_names]
        
        for name_parts in names:
            assert isinstance(name_parts, list)
            n = len(name_parts)
            for i in range(n):
                for f_i, value in enumerate(
                    extract_features(name_parts, i, n, self._feature_characters)
                ):
                    feature_values[f_i].append(value)

        df_test = pd.DataFrame().reindex(columns=self._X.columns)
        for i in range(len(feature_names)):
            if feature_names[i] == 'stem':
                for known_stem_value in self._stem_values:
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

    def predict_one(self, name_parts):
        """
        Классифицировать части имени
        list[name_part] -> list[tag]
        """
        assert isinstance(name_parts, list)
        X_test = self._build_X_predict([name_parts])
        return list(self._classifier.predict(X_test))
    
    def predict(self, names):
        """
        Классифицировать части имени для множества имён
        list[list[name_part]] -> typing.Iterable[list[tag]]
        """
        assert isinstance(names, list)
        X_test = self._build_X_predict(names)
        y_pred = self._classifier.predict(X_test)
        i = 0
        for name_parts in names:
            n = len(name_parts)
            yield list(y_pred[i : i + n])
            i += n


# %%
facade_classifier = FacadeClassifier()
facade_classifier.fit(root)

predicted_tags = facade_classifier.predict(
    [
        ['Mauro', 'Camoranezi', 'Jr'],
        ['Cathrine', 'III'],
    ]
)
list(predicted_tags)

# %% [markdown]
# ### Измерим accuracy при классификации имени целиком

# %%
import numpy as np
from sklearn.model_selection import KFold


def get_k_fold_indices(name_tags, *, n_splits=5, shuffle=False, random_state=None):
    assert isinstance(n_splits, int)
    assert isinstance(shuffle, bool)
    indices = np.linspace(0, len(root), num=len(root), endpoint=False, dtype='int')
    k_fold = KFold(n_splits, shuffle=shuffle)
    return k_fold.split(indices)


def cross_validate_facade_classifier(
    facade_classifier,
    name_tags,
    *,
    n_splits=5,
    shuffle=False,
    random_state=None,
):
    scores = []
    for train, test in get_k_fold_indices(
        name_tags,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    ):
        train_name_tags = [name_tags[i] for i in train]
        facade_classifier.fit(train_name_tags)

        num_correct_pred = 0
        num_failed_pred = 0
        names = [
            [
                name_part_tag.text for name_part_tag in name_tags[i]
            ]
            for i in test
        ]
        actual_tags = [
            [
                name_part_tag.tag for name_part_tag in name_tags[i]
            ]
            for i in test
        ]
        predicted_tags = list(facade_classifier.predict(names))

        for i in range(len(test)):
            if actual_tags[i] == predicted_tags[i]:
                num_correct_pred += 1
            else:
                num_failed_pred += 1
            # print(names[i])
            # print(actual_tags[i])
            # print(predicted_tags[i])
        scores.append(num_correct_pred / (num_correct_pred + num_failed_pred))
    return scores

scores = cross_validate_facade_classifier(FacadeClassifier(), root, shuffle=True)
scores

# %% [markdown]
# # Классификация набора токенов 

# %%
[0, 1, 2, 3][0: 2]
