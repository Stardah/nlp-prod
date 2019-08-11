from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
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
            ('clf', SGDClassifier(loss='log', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=5, tol=None)),
        ])

    def fit(self, X_train, y_train):
        self.text_clf.fit(X_train, y_train)

    def predict(self, X_test):
        self.predicted = self.text_clf.predict(X_test)
        return self.predicted

    def predict_proba(self, X_test):
        self.proba = self.text_clf.predict_proba(X_test)
        return self.proba