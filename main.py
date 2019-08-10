import pandas as pd
import numpy as np
import re
import math
import collections as co
import scipy
from scipy import stats

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

# Use nltk for valid words
import nltk
import collections as co
import scipy

from collections import Counter
from math import sqrt


def write_answer(file_name, answer):
    #file = open('./results/' + str(file_name) + '.txt', 'w')
    #file.write(str(answer))
    #file.close()
    print('woob woob')


write_answer('self0', 'example_answer')

df = pd.read_csv('./df.csv')
df = df.sample(10000)

# оставим в выборке только положительние или отрицательные текста
df = df[df['sentiment'] > 0]

df['text_arr'] = df.text.str.split('\W+').tolist()

# сделаем массив из всех слов, разделенных пробелами, которые могут встречаться
text_arrays = df['text_arr'].tolist()
text_arrays = np.concatenate(text_arrays, axis=0)

# уберем все пустые строки
# подсказка функция filter
text_arrays = list(text_arrays)
text_arrays = np.array(list(filter(lambda s: s != '', text_arrays)))

# очистим от мусорных слов
# pip install stop_words

#from stop_words import get_stop_words as gsw2

stopWords = nltk.corpus.stopwords.words()
#sw2 = gsw2('ru')
#stopWords.extend(sw2)
stopWords.extend(['http', 'rt', 'числ', 'со', 'co'])

filter_df = pd.DataFrame({'words': text_arrays})
stop_df = pd.DataFrame({'words_stop': stopWords})

filter_df = pd.merge(filter_df
                     , stop_df
                     , how='left'
                     , left_on='words'
                     , right_on='words_stop'
                     )[:]

filter_df = filter_df[(filter_df.words_stop.isnull())][:]
filter_df.drop(['words_stop'], axis=1, inplace=True)

df['cnt_yes'] = df['text'].str.count('\Wда\W')

filter_df['cnt_letter'] = filter_df.words.str.len()
filter_df['non_letter'] = filter_df.words.str.contains(r'^\W+$')
filter_df['is_number'] = filter_df.words.str.contains(r'число')

# давайте обогатим наши стоп слова по правилу если меньше 3 букв или нет букв вообще относим к стоп словам
new_stop_words = filter_df[(filter_df.cnt_letter < 2)
                           | (filter_df.non_letter == True)
                           | (filter_df.is_number == True)
                           ].words.values

stopWords.extend(new_stop_words)

df['text_arr'] = df.text.str.split('\W+').tolist()

# делаем стемым только из слов не из стоп листа
from nltk.stem.snowball import RussianStemmer

stemming = RussianStemmer()


def stem_list(row):
    my_list = row['text_arr']
    stemmed_list = [stemming.stem(word) if word not in stopWords else '' for word in my_list]
    return (stemmed_list)


df['stem_arr'] = df.apply(stem_list, axis=1)

text_arrays = df['stem_arr'].tolist()
text_arrays = np.concatenate(text_arrays, axis=0)

# уберем все пустые строки
text_arrays = list(text_arrays)
text_arrays = list(filter(None, text_arrays))

c_text = co.Counter(text_arrays)
# оставим для модели топ 500 слов
final_words = pd.DataFrame.from_dict(dict(c_text), orient='index').sort_values(by=0, ascending=False).head(
    500).index.values

f = lambda x: ' '.join([item for item in x if item in final_words])
df['stem_text'] = df['stem_arr'].apply(f)

df.drop(['text', 'text_arr'], axis=1, inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer.fit(df.stem_text)
df_tfidf = vectorizer.transform(df.stem_text)
df_tfidf = pd.DataFrame(df_tfidf.todense())

# make final df
model_df = pd.merge(df.reset_index(drop=True)[:],
                    df_tfidf,
                    how='inner',
                    left_index=True,
                    right_index=True)

interest_columns = list(set(model_df.columns) - set(['sentiment', 'stem_arr', 'stem_text']))
model_df['sentiment'] = [1 if x == 1 else 0 for x in model_df['sentiment']]
X_train, X_test, y_train, y_test = train_test_split(model_df[interest_columns], model_df['sentiment'], test_size=0.3,
                                                    random_state=42)

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import lightgbm as lgb

params = {
    # 'bagging_fraction': 0.1,
    # 'bagging_freq': 2,
    'boosting_type': 'gbdt',
    # 'colsample_bytree': 0.1,
    'feature_fraction': 0.3,
    'learning_rate': 0.09,
    'max_depth': -1,
    # 'num_leaves': 32,
    'reg_lambda': 0.01,
    'subsample': 0.2,
    'objective': 'binary',
    'metric': 'auc'}

lgb_data_train = lgb.Dataset(X_train[interest_columns],
                             y_train,
                             free_raw_data=False
                             )
model = lgb.train(params, lgb_data_train, num_boost_round=100)

from sklearn.metrics import accuracy_score

test_acc = accuracy_score(y_test, model.predict(X_test).round())
print(test_acc)
write_answer('self0', test_acc)
