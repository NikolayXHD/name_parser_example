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
# ### Часть 3. Проверка качества перевода.
#
# Ниже - случайный текст из Википедии в оригинале и его переводы в двух МТ: Google Translate и DeepL.
#
# Предположим, что один из этих переводов - машинный перевод (`mt`), а второй - постредактура переводчиком (`edit`).
#
# **Задача:** На основе этих двух текстов посчитать, как сильно постредактура изменила машинный перевод. Фактически - показать, насколько машинный перевод хороший/плохой.
#
# **NB!** Можно использовать, например, разные метрики для оценки качества перевода, типа BLEU и посложнее.
#
# Сильно заморачиваться не стоит - представь, что нужно asap общее понимание качества МТ, некая метрика. И будет здорово кратко обосновать, почему эта метрика достаточна для общего понимания качества перевода.

# %%
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

# %%
# !pip install nltk

# %%
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# %%
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score

def get_meteor_score(reference, hypothesis):
    ref_tokens = word_tokenize(reference)
    hyp_tokens = word_tokenize(hypothesis)
    return meteor_score.meteor_score([ref_tokens], hyp_tokens)

score = get_meteor_score(edited, mt)
print(score)

# %% [markdown]
# ### Почему meteor_score, и почему этого достаточно
#
# - Имеет более высокую корреляцию с человеческой оценкой качества, чем bleu. Насколько мне стало известно из Википедии, драматических улучшений в степени корреляции с человеческой оценкой по сравнению с bleu не достигнуто, так что утверждение "метрика несполько лучше коррелирует, чем bleu" равносильно "метрика одна из лучших по степени корреляции на данный момент".
# - В отличие от bleu, учитывает не только precision, но и recall, т.е. потеря части информации по сравнению с эталонным переводом
# - В отличие от bleu, учитывает синонимы
# - Учитывает совпадение порядка слов
# - есть готовая реализация, не нужно писать самому
