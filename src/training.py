import os
import time
import multiprocessing
import dill

import hyperopt as hpo

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
# import catboost as cat

from . import pipeline_utils as pipe



class HyperoptTrainer:
    def __init__(self,
                 model_type,
                 model_name,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 *,
                 models_path='models',
                 seed=42,
                 prints=False,
                 importance=None,
                 log_transform=False,
                 **model_kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

        self.models_path = models_path
        self.seed = seed
        self.prints = prints
        self.importance = importance
        self.model_kwargs = model_kwargs

        self.log_transform = log_transform

        self.trials = hpo.Trials()
        self.log = None

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return dill.load(f)

    def make_xgb_model(self):
        for param in ['n_estimators', 'max_depth', 'num_parallel_tree']:
            if param in self.hyperparams:
                self.hyperparams[param] = int(self.hyperparams[param])

        self.model = xgb.XGBRegressor(
            random_state=self.seed,
            **self.hyperparams,
            **self.model_kwargs,
        )

    def make_rf_model(self):
        for param in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_leaf_nodes', ]:
            if param in self.hyperparams and self.hyperparams[param] is not None:
                self.hyperparams[param] = int(self.hyperparams[param])
         
        self.model = RandomForestRegressor(
            random_state=self.seed,
            **self.hyperparams,
            **self.model_kwargs,
        )

    def make_lgb_model(self):
        for param in ['n_estimators', 'max_depth', 'num_iterations', 'num_leaves', 'subsample_freq', 'min_child_samples']:
            if param in self.hyperparams:
                self.hyperparams[param] = int(self.hyperparams[param])
         
        self.model = lgb.LGBMRegressor(
            random_state=self.seed,
            **self.hyperparams,
            **self.model_kwargs,
        )

    def objective(self, space):
        self.hyperparams = dict(sorted(list(space.items()), key=lambda x: x[0]))
        if self.prints:
            print(self.hyperparams)
        if self.log is None:
            self.log = pd.DataFrame(columns=['timestamp', 'time', 'train_score', 'test_score', 'hyperparam_dict'] + sorted(list(self.hyperparams.keys())))
        space_values = list(self.hyperparams.values())

        if self.model_type == 'xgb':
            self.make_xgb_model()
        elif self.model_type == 'rf':
            self.make_rf_model()
        elif self.model_type == 'lgb':
            self.make_lgb_model()
        else:
            raise 'Unsupported model'

        if 'num_feats' in self.hyperparams is not None and self.importance is not None:
            self.model = Pipeline(steps=[
                ('column_importance_selector', pipe.column_importance_selector(int(self.hyperparams.pop('num_feats')), self.importance)),
                ('model', self.model),
            ])

        tic = time.time()

        self.train()

        toc = time.time()

        train_score = self.results['scores']['rmsle_train']
        test_score = self.results['scores']['rmsle_test']

        best = False
        if self.log is not None and len(self.log) > 0 and test_score < self.log['test_score'].min():
            best = True

        self.log.loc[self.log.index.max() + 1 if len(self.log) > 0 else 0] = [time.strftime('%d-%m-%Y %H:%M:%S'),
                                                                              (toc - tic)/60,
                                                                              train_score,
                                                                              test_score,
                                                                              self.hyperparams] + space_values
        self.save(last=True, best=best)
        return test_score

    def fmin(self, space, max_evals=500):
        best = hpo.fmin(fn=self.objective,
                        space=space,
                        algo=hpo.tpe.suggest,
                        max_evals=max_evals,
                        trials=self.trials,
                        rstate=np.random.seed(self.seed),
                        verbose=True)

    def get_path(self, best=True):
        if best:
            return os.path.join(self.models_path, f'model_{self.model_name}_best.pkl')
        else:
            return os.path.join(self.models_path, f'model_{self.model_name}_last.pkl')
        
    def save(self, last=False, best=False):
        if best:
            with open(self.get_path(best=True), 'wb') as f:
                dill.dump(self, f)
        if last:
            with open(self.get_path(best=False), 'wb') as f:
                dill.dump(self, f)

    def train(self):
        if self.log_transform:
            self.model.fit(self.X_train, self.y_train.apply(np.log1p))
        else:
            self.model.fit(self.X_train, self.y_train)

        preds_train = pd.Series(self.model.predict(self.X_train), index=self.X_train.index)
        preds_test = pd.Series(self.model.predict(self.X_test), index=self.X_test.index)

        if self.log_transform:
            preds_train = preds_train.apply(np.expm1)
            preds_test = preds_test.apply(np.expm1)
            
        preds_train = preds_train.apply(lambda x: max(x, 0.0))
        preds_test = preds_test.apply(lambda x: max(x, 0.0))

        self.results = dict(
            scores=dict(
                rmsle_train=np.sqrt(metrics.mean_squared_log_error(self.y_train, preds_train)),
                rmsle_test=np.sqrt(metrics.mean_squared_log_error(self.y_test, preds_test)),
            ),
            preds=dict(
                preds_train=preds_train,
                preds_test=preds_test,
            ),
        )
    
    def plot(self, feat='test_score', log=None):
        if log is None:
            log = self.log
        for col in log.columns.difference(['timestamp', 'time', 'train_score', 'test_score', 'hyperparam_dict']):
            plt.scatter(log[col], log[feat])
            plt.title(col)
            plt.show()





