from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append((classifier[0],classifier[1].predict_proba(X).tolist()))

        return self.predictions_
