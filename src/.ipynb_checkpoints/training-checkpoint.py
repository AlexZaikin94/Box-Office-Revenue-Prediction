import os
import time
import dill

import hyperopt as hpo

import numpy as np
import pandas as pd

from sklearn import metrics

import xgboost as xgb
import lightgbm as lgb
import catboost as cat



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
        self.model_kwargs = model_kwargs

        self.trials = hpo.Trials()
        self.log = None
        self.min_v = self.y_train.min()
        self.max_v = self.y_train.max()

    def make_xgb_model(self, hyperparams):
        n_estimators =      int(2 ** hyperparams.pop('n_estimators'))
        max_depth =         int(hyperparams.pop('max_depth'))
        num_parallel_tree = int(hyperparams.pop('num_parallel_tree'))

        learning_rate = 10 ** -(hyperparams.pop('learning_rate')/2)
        gamma =         10 ** -(hyperparams.pop('gamma')/2)
        reg_lambda =    10 ** -(hyperparams.pop('reg_lambda')/2)

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            gamma=gamma,
            reg_lambda=reg_lambda,
            max_depth=max_depth,
            num_parallel_tree=num_parallel_tree,
            random_state=self.seed,
            **hyperparams,
            **self.model_kwargs,
        )

    def make_cat_model(self, hyperparams):
        self.model = cat.CatBoostRegressor(
            **hyperparams,
            **self.model_kwargs,
        )

    def make_lgb_model(self, hyperparams):
        self.model = lgb.LGBMRegressor(
            **hyperparams,
            **self.model_kwargs,
        )

    def objective(self, space):
        space = dict(sorted(list(space.items()), key=lambda x: x[0]))
        if self.log is None:
            self.log = pd.DataFrame(columns=['timestamp', 'time', 'train_score', 'test_score', 'hyperparam_dict'] + sorted(list(space.keys())))
        space_values = list(space.values())

        if self.model_type == 'xgb':
            self.make_xgb_model(space)
        elif self.model_type == 'cat':
            self.make_cat_model(space)
        elif self.model_type == 'lgb':
            self.make_lgb_model(space)
        else:
            raise 'Unsupported model'

        tic = time.time()
        results = self.train()
        toc = time.time()

        train_score = results['scores']['rmsle_train']
        test_score = results['scores']['rmsle_test']

        best = False
        if self.log is not None and len(self.log) > 0 and test_score < self.log['test_score'].min():
            best = True
        self.save(last=True, best=best)

        self.log.loc[self.log.index.max() + 1 if len(self.log) > 0 else 0] = [time.strftime('%d-%m-%Y %H:%M:%S'),
                                                                              (toc - tic)/60,
                                                                              train_score,
                                                                              test_score,
                                                                              space] + space_values
        return test_score

    def fmin(self, space, max_evals=500):
        best = hpo.fmin(fn=self.objective,
                        space=space,
                        algo=hpo.tpe.suggest,
                        max_evals=max_evals,
                        trials=self.trials,
                        rstate=np.random.seed(self.seed),
                        verbose=True)

    def save(self, last=False, best=False):
        if best:
            with open(os.path.join(self.models_path, f'model_{self.model_name}_best.pkl'), 'wb') as f:
                dill.dump(self, f)
        if last:
            with open(os.path.join(self.models_path, f'model_{self.model_name}_last.pkl'), 'wb') as f:
                dill.dump(self, f)
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)

        preds_train = pd.Series(self.model.predict(self.X_train),
                                index=self.X_train.index).apply(lambda x: min(self.max_v, max(x, self.min_v)))
        preds_test = pd.Series(self.model.predict(self.X_test),
                               index=self.X_test.index).apply(lambda x: min(self.max_v, max(x, self.min_v)))

        results = dict(
            scores=dict(
                rmsle_train=np.sqrt(metrics.mean_squared_log_error(self.y_train, preds_train)),
                rmsle_test=np.sqrt(metrics.mean_squared_log_error(self.y_test, preds_test)),
            ),
            preds=dict(
                preds_train=preds_train,
                preds_test=preds_test,
            ),
        )
        if self.prints:
            print(f"train RMSLE: {results['scores']['rmsle_train']:.3f}")
            print(f"test RMSLE: {results['scores']['rmsle_test']:.3f}")

        return results





