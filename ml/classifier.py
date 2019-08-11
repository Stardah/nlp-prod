from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline


class TextCF:
    """
        Класс обёртка для пайплайна обработки и классификации текстовых данных
        (когда-то был более полным и сложным, но в итоге остался просто обёрткой)
    """

    def __init__(self):
        self.text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',  LogisticRegression(random_state=0, solver='lbfgs')),
        ])

    def fit(self, X_train, y_train):
        self.text_clf.fit(X_train, y_train)

    def predict(self, X_test):
        self.predicted = self.text_clf.predict(X_test)
        return self.predicted

    def predict_proba(self, X_test):
        self.proba = self.text_clf.predict_proba(X_test)
        return self.proba


class FeatureGen():
    def fit(self, X, y):
        return self

    def transform(self, X):
        import pandas as pd
        features = pd.DataFrame(X.str.count('\w+-?\w+').values, columns=['cnt'])
        features['cnt_!'] = X.str.count('\!').values
        features['cnt_?'] = X.str.count('\?').values
        features['cnt_.'] = X.str.count('\.').values
        return features