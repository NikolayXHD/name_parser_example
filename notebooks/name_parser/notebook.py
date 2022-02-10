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
        return tuple(self._classifier.predict(X_test))
    
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
            yield tuple(y_pred[i : i + n])
            i += n


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
            tuple(
                name_part_tag.tag for name_part_tag in name_tags[i]
            )
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
print(scores)
print(f'{np.array(scores).mean():.3f} ± {np.array(scores).std():.3f}')

# %% [markdown]
# ## Классификация с учётом распределения паттернов в обучающей выборке
#
# Под паттерном подразумевается порядок частей 1 имени, например
# ```
# ('GivenName', 'Surname', 'SuffixGenerational')
# ```

# %%
from collections import Counter, defaultdict

class FacadeClassifierWholistic(FacadeClassifier):
    """
    Классификатор для имени целиком: list[name_part] -> list[tag]
    В отличие от родительского класса, учитывает распределение наборов тегов
    в обучающей выборке.

    Под набором тегов подразумевается теги всех частей имени
    """
    def fit(self, name_tags):
        super().fit(name_tags)
        num_to_pattern_to_count = defaultdict(Counter)
        num_name_parts = 0
        for name_tag in name_tags:
            assert name_tag.tag == 'Name'
            classes = tuple(
                name_part_tag.tag 
                for name_part_tag in name_tag
            )
            num_to_pattern_to_count[len(classes)].update([classes])
            num_name_parts += len(name_tag)
            
        self._num_to_pattern_to_count = num_to_pattern_to_count
        self._num_patterns = len(name_tags)
        self._num_name_parts = num_name_parts
        self._class_to_index = {
            klass: i
            for i, klass in enumerate(self._classifier.classes_)
        }
    
    def predict_one(self, name_parts):
        """
        Классифицировать части имени
        list[name_part] -> list[tag]
        """
        assert isinstance(name_parts, list)
        X_test = self._build_X_predict([name_parts])
        # Согласно описанию алгоритмов NaiveBayes в документации scikit-learn
        # на predict_proba не стоит полагаться.
        # Здесь мы осознанно пренебрегаем этим, полагаясь на кросс-валидацию
        # с измерением accuracy
        y_pred_proba = self._classifier.predict_proba(X_test)
        return self._choose_pattern(y_pred_proba)
    
    def predict(self, names):
        """
        Классифицировать части имени для множества имён
        list[list[name_part]] -> typing.Iterable[list[tag]]
        """
        assert isinstance(names, list)
        X_test = self._build_X_predict(names)
        y_pred = self._classifier.predict_proba(X_test)
        i = 0
        for name_parts in names:
            n = len(name_parts)
            yield self._choose_pattern(y_pred[i : i + n,:])
            i += n
    
    def _choose_pattern(self, y_pred_proba):
        num_name_parts = y_pred_proba.shape[0]
        pattern_to_count = self._num_to_pattern_to_count[num_name_parts]
        pattern_to_proba = Counter()
        for pattern, count in pattern_to_count.items():
            proba = count
            for i, part_class in enumerate(pattern):
                class_index = self._class_to_index[part_class]
                proba *= y_pred_proba[i][class_index]
            pattern_to_proba[pattern] = proba
        obvious_pattern = tuple(
            self._classifier.classes_[y_pred_proba[i].argmax()]
            for i in range(num_name_parts)
        )
        if obvious_pattern not in pattern_to_proba:
            # примем за ожидаемую частоту паттрна, которого нет в 
            # обучающей выборке размера N
            # P = 0.5 * 1 / N
            proba = 0.5
            for i in range(num_name_parts):
                proba *= y_pred_proba[i].max()
            pattern_to_proba[obvious_pattern] = proba

        return pattern_to_proba.most_common(1)[0][0]


# %% [markdown]
# ### Оценим accuracy

# %%
scores = cross_validate_facade_classifier(FacadeClassifierWholistic(), root, shuffle=True)
print(scores)
print(f'{np.array(scores).mean():.3f} ± {np.array(scores).std():.3f}')

# %% [markdown]
# По сравнению с предыдущим результатом, без учёта распределения паттернов, результат не улучшился.

# %% [markdown]
# # Ниже можно вручную взаимодейсвовать с классификаторами

# %%
facade_classifier_wholistic = FacadeClassifierWholistic()
facade_classifier_wholistic.fit(root)

facade_classifier = FacadeClassifier()
facade_classifier.fit(root)

# %% [markdown]
# ### Классификатор с учётом распределения паттернов

# %%
predicted_tags = facade_classifier_wholistic.predict(
    [
        ['Mauro', 'Camoranezi', 'Jr'],
        ['Cathrine', 'III'],
    ]
)
for prediction in predicted_tags:
    print(prediction)

# %% [markdown]
# ### Классификатор, учитывающий признаки отдельных токенов

# %%
predicted_tags = facade_classifier.predict(
    [
        ['Mauro', 'Camoranezi', 'Jr'],
        ['Cathrine', 'III'],
    ]
)
for prediction in predicted_tags:
    print(prediction)


# %% [markdown]
# # Поддержим вывод, совпадающий с исходным форматом XML

# %%
def parse(lines, classifier):
    names = [line.split(' ') for line in lines.split('\n')]
    predictions = list(classifier.predict(names))
    return '\n'.join(
        ' '.join(
            f'<{predictions[j][i]}>{names[j][i]}</{predictions[j][i]}>'
            for i in range(len(names[j]))
        )
        for j in range(len(names))
    )


# %%
output = parse(
    'Mr. Alexey Konstantinovich Tolstoy Sr\n'
    'Mrs Anna Karenina',
    facade_classifier_wholistic
)
print(output)
