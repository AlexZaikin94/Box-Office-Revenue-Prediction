import re
import math

import numpy as np
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from . import utils



def column_selector(X, feats=None):
    return X[feats]



def column_importance_selector(num_feats, importance):
    def column_importance_select(X, num_feats, importance):
        importance = importance.sort_values(key=lambda x: abs(x), ascending=False)
        feats = list(importance.index[:num_feats])
        return X[feats].copy()
    return FunctionTransformer(column_importance_select, kw_args={'num_feats': num_feats, 'importance': importance})

def column_dropper(X, feats=None):
    return X.drop(columns=feats)


def evaluator(X, feats):
    for feat in feats:
        X.loc[~X[feat].isna(), feat] = X.loc[~X[feat].isna(), feat].apply(eval)
    return X


def special_char_remover(X):
    return X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


class Json_one_hot_encoder:
    def __init__(self, feats, top=100, threshold=30, verbose=False):
        self.feats = feats
        self.stats = dict.fromkeys(self.feats.keys(), None)
        self.values = dict.fromkeys(self.feats.keys(), None)
        self.top = top
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        for feat, cols in self.feats.items():
            self.stats[feat] = dict()
            self.values[feat] = dict()
            print(feat) if self.verbose else None
            temp_feat_X = X[feat].apply(pd.Series).stack()
            for col in cols:
                print(' ' + col) if self.verbose else None
                self.stats[feat][col] = temp_feat_X.apply(lambda x: x[col]).value_counts().sort_values(ascending=False)
                self.values[feat][col] = list(self.stats[feat][col][self.stats[feat][col] > self.threshold].head(self.top).index)
#         for feat, cols in self.feats.items():
#             for col in cols:
#                 self.stats[feat][col] = X[feat].apply(pd.Series).stack().apply(lambda x: x[col]).value_counts().sort_values(ascending=False)
#                 temp = self.stats[feat][col]
#                 self.values[feat][col] = list(temp[temp > self.threshold].head(self.top).index)
        return self

    def transform(self, X, y=None):
        for feat, cols in self.feats.items():
            for col in cols:
                X_feat_col = X[feat].apply(lambda x: [i[col] for i in x]).apply(lambda x: dict(zip(*np.unique(x, return_counts=True))))
                for val in self.values[feat][col]:
                    X[feat + '_' + col + '_' + str(val)] = X_feat_col.apply(lambda x: x.get(val, 0))
                X[feat + '_' + col + '_unique'] = X_feat_col.apply(len)
            X_feat_len = X[feat].apply(len)
#             if not all(X_feat_len.values == X[feat + '_' + col + '_unique'].values):
            X[feat + '_len'] = X_feat_len
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


month_cos = lambda x: math.cos(2 * math.pi * int(x[5:7]) / 12)
month_sin = lambda x: math.sin(2 * math.pi * int(x[5:7]) / 12)


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
            X[feat + '_month_cos'] = X[feat].apply(lambda x: month_cos(x))
            X[feat + '_month_sin'] = X[feat].apply(lambda x: month_sin(x))
        return X


