import re

import numpy as np
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def column_selector(X, feats=None):
    return X[feats]


def column_dropper(X, feats=None):
    return X.drop(columns=feats)


def evaluator(X, feats):
    for feat in feats:
        X.loc[~X[feat].isna(), feat] = X.loc[~X[feat].isna(), feat].apply(eval)
    return X


def special_char_remover(X):
    return X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


class Json_one_hot_encoder:
    def __init__(self, feats, top=100, threshold=30):
        self.feats = feats
        self.stats = dict.fromkeys(self.feats.keys(), None)
        self.values = dict.fromkeys(self.feats.keys(), None)
        self.top = top
        self.threshold = threshold

    def fit(self, X, y=None):
        for feat, col in self.feats.items():
            self.stats[feat] = X[feat].apply(pd.Series).stack().apply(lambda x: x[col]).value_counts().sort_values(ascending=False)
            temp = self.stats[feat]
            self.values[feat] = list(temp[temp > self.threshold].head(self.top).index)
        return self

    def transform(self, X, y=None):
        for feat, col in self.feats.items():
            X[feat] = X[feat].apply(lambda x: [i[col] for i in x if i[col]])
            for val in self.values[feat]:
                X[feat + '_' + str(val)] = X[feat].apply(lambda x: int(val in x))
            X[feat + '_len'] = X[feat].apply(len)
        return X


class One_hot_encoder:
    def __init__(self, feats, top=100, threshold=30):
        self.feats = feats
        self.stats = dict.fromkeys(self.feats, None)
        self.values = dict.fromkeys(self.feats, None)
        self.top = top
        self.threshold = threshold

    def fit(self, X, y=None):
        for feat in self.feats:
            self.stats[feat] = X[feat].value_counts().sort_values(ascending=False)
            temp = self.stats[feat]
            self.values[feat] = list(temp[temp > self.threshold].head(self.top).index)
        return self

    def transform(self, X, y=None):
        for feat in self.feats:
            for val in self.values[feat]:
                X[feat + '_' + str(val)] = X[feat].apply(lambda x: int(x == val))
        return X


class Imputer:
    def __init__(self, feats, strategy):
        self.feats = feats
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.feats])
        return self

    def transform(self, X, y=None):
        X.loc[:, self.feats] = self.imputer.transform(X[self.feats])
        return X


class Date_transformer:
    def __init__(self, feats):
        self.feats = feats
        self.max_year = dict.fromkeys(self.feats, None)

    def fit(self, X, y=None):
        for feat in self.feats:
            self.max_year[feat] = X[feat].apply(lambda x: int(x[:4])).max()
        return self

    def transform(self, X, y=None):
        for feat in self.feats:
            X[feat + '_year'] = self.max_year[feat] - X[feat].apply(lambda x: int(x[:4]))
        return X


