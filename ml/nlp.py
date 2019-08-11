import collections as co

import lightgbm as lgb
import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from ml.classifier import TextCF


class NLP_model():

    def __init__(self):
        print('woob woob')
        df = pd.read_csv('./data/sent_df_prep.csv')
        #df['sentiment'] = [1 if x == 1 else 0 for x in df['sentiment']]
        #df = self.preprocess_data(df)
        self.fit_model(df)

    def preprocess_data(self, df):
        df['text'] = df['text'].str.replace("\n", " ")

        df['text'] = df['text'].str.lower()

        # Добавим фактор количество слов в предложении
        df['cnt_words'] = df['text'].str.count('\w+-?\w+')

        # Удалим все ники с учетом первого символа @
        df['text'] = df['text'].str.replace(r'\B@[\w\d]+\s?', '')

        # Превратим двойные пробелы в одинарные
        df['text'] = df['text'].apply(lambda t: t.replace('  ', ' '))

        # Превратите все числа в фразу "число"
        df['text'] = df['text'].str.replace(r'\b\d+\b', "число", regex=True)

        # Соединим не и слово с помощью замены "не " на "не_"
        df['text'] = df['text'].str.replace(r'\bне\s\b', "не_", regex=True)

        df['cnt_!'] = df['text'].str.count('\!')
        df['cnt_?'] = df['text'].str.count('\?')
        df['cnt_.'] = df['text'].str.count('\.')
        return df

    def fit_model(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.3, random_state=42)
        textCF = TextCF()
        textCF.fit(X_train, y_train)
        self.model = textCF

    def define_stop_words(self, text_arrays):
        # from stop_words import get_stop_words as gsw2
        stop_words = nltk.corpus.stopwords.words()
        # sw2 = gsw2('ru')
        # stopWords.extend(sw2)
        stop_words.extend(['http', 'rt', 'числ', 'со', 'co'])

        filter_df = pd.DataFrame({'words': text_arrays})
        stop_df = pd.DataFrame({'words_stop': stop_words})

        filter_df = pd.merge(filter_df
                             , stop_df
                             , how='left'
                             , left_on='words'
                             , right_on='words_stop'
                             )[:]

        filter_df = filter_df[(filter_df.words_stop.isnull())][:]
        filter_df.drop(['words_stop'], axis=1, inplace=True)

        filter_df['cnt_letter'] = filter_df.words.str.len()
        filter_df['non_letter'] = filter_df.words.str.contains(r'^\W+$')
        filter_df['is_number'] = filter_df.words.str.contains(r'число')

        # давайте обогатим наши стоп слова по правилу если меньше 3 букв или нет букв вообще относим к стоп словам
        new_stop_words = filter_df[(filter_df.cnt_letter < 2)
                                   | (filter_df.non_letter == True)
                                   | (filter_df.is_number == True)
                                   ].words.values

        stop_words.extend(new_stop_words)
        self.stop_words = stop_words

    def prepare_dataset(self, samples=10000):
        df = pd.read_csv('./data/df.csv')
        df = df.sample(10000)

        # оставим в выборке только положительние или отрицательные текста
        df = df[df['sentiment'] > 0]

        df['text_arr'] = df.text.str.split('\W+').tolist()
        df['cnt_yes'] = df['text'].str.count('\Wда\W')

        # сделаем массив из всех слов, разделенных пробелами, которые могут встречаться
        text_arrays = df['text_arr'].tolist()
        text_arrays = np.concatenate(text_arrays, axis=0)

        # уберем все пустые строки
        # подсказка функция filter
        text_arrays = list(text_arrays)
        text_arrays = np.array(list(filter(lambda s: s != '', text_arrays)))

        # очистим от мусорных слов
        # pip install stop_words

        # efwefwef
        self.define_stop_words(text_arrays)

        stemming = RussianStemmer()

        df['stem_arr'] = df.apply(
            lambda x: [stemming.stem(word) if word not in self.stop_words else '' for word in x['text_arr']], axis=1)

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

        vectorizer = TfidfVectorizer()
        vectorizer.fit(df.stem_text)
        df_tfidf = vectorizer.transform(df.stem_text)
        df_tfidf = pd.DataFrame(df_tfidf.todense())

        # make final df
        self.model_df = pd.merge(df.reset_index(drop=True)[:],
                                 df_tfidf,
                                 how='inner',
                                 left_index=True,
                                 right_index=True)

    def fit(self):
        interest_columns = list(set(self.model_df.columns) - set(['sentiment', 'stem_arr', 'stem_text']))
        self.model_df['sentiment'] = [1 if x == 1 else 0 for x in self.model_df['sentiment']]
        X_train, X_test, y_train, y_test = train_test_split(self.model_df[interest_columns],
                                                            self.model_df['sentiment'], test_size=0.3,
                                                            random_state=42)
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
                                     y_train, free_raw_data=False)
        self.model = lgb.train(params, lgb_data_train, num_boost_round=100)

    def predict(self, x):
        #df = pd.DataFrame(x, columns=['text'])
        return self.model.predict(x)
