import os
import time
import dill

import hyperopt as hpo

import numpy as np
import pandas as pd

from sklearn import metrics#, mean_squared_error, mean_squared_log_error

import xgboost as xgb
import lightgbm as lgb
import catboost as cat



def train(model, X_train, y_train, X_test, y_test, prints=False):
    model.fit(X_train, y_train)
    min_v = y_train.min()
    max_v = y_train.max()

    preds_train = pd.Series(model.predict(X_train), index=X_train.index).apply(lambda x: min(max_v, max(x, min_v)))
    preds_test = pd.Series(model.predict(X_test), index=X_test.index).apply(lambda x: min(max_v, max(x, min_v)))

    results = dict(
        scores=dict(
            rmsle_train=np.sqrt(metrics.mean_squared_log_error(y_train, preds_train)),
            rmsle_test=np.sqrt(metrics.mean_squared_log_error(y_test, preds_test)),
#             rmse_train=np.sqrt(metrics.mean_squared_error(y_train, preds_train)),
#             rmse_test=np.sqrt(metrics.mean_squared_error(y_test, preds_test)),
        ),
        preds=dict(
            preds_train=preds_train,
            preds_test=preds_test,
        ),
    )
    if prints:
        print(f"train RMSLE: {results['scores']['rmsle_train']:.3f}")
        print(f"test RMSLE: {results['scores']['rmsle_test']:.3f}")
#         print('train RMSE:', results['scores']['rmse_train'])
#         print('test RMSE:', results['scores']['rmse_test'])
    
    return model, results


def corr(X, y):
    _X = X.copy()
    _X['y'] = y
    corr_vals = _X.corr()['y'].sort_values(key=lambda x: abs(x), ascending=False)
    return pd.DataFrame(corr_vals.drop('y'))


class HyperoptTrainer:
    def __init__(self, model_type, model_name, X_train, y_train, X_test, y_test, models_path='models', seed=42, prints=False, **kwargs):
        self.model_type = model_type
#         self.model_objective = model_objective
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
        self.models_path = models_path
        self.seed = seed
        self.prints = prints
        self.kwargs = kwargs

        self.trials = hpo.Trials()
        self.log = None

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
            **hyperparams,
        )

    def make_cat_model(self, hyperparams):
        self.model = cat.CatBoostRegressor(
            **hyperparams,
        )

    def make_lgb_model(self, hyperparams):
        self.model = lgb.LGBMRegressor(
            **hyperparams,
        )

    def objective(self, space):
        space = dict(sorted(list(space.items()), key=lambda x: x[0]))
        if self.log is None:
            self.log = pd.DataFrame(columns=['timestamp', 'time', 'train_score', 'test_score', 'hyperparam_dict'] + sorted(list(space.keys())))
        space_values = list(space.values())
        model_type = space.pop('model_type')

        if self.model_type == 'xgb':
            self.make_xgb_model(space)
        elif self.model_type == 'cat':
            self.make_cat_model(space)
        elif self.model_type == 'lgb':
            self.make_lgb_model(space)
        else:
            raise 'Unsupported model'

        tic = time.time()
        self.model, results = train(self.model, self.X_train, self.y_train, self.X_test, self.y_test, prints=self.prints)
        train_score = results['scores']['rmsle_train']
        test_score = results['scores']['rmsle_test']
        
        if self.log is not None:
            if len(self.log) > 0:
                if test_score < self.log['test_score'].min():
                    self.save()
            else:
                self.save()

        self.log.loc[self.log.index.max() + 1 if len(self.log) > 0 else 0] = [time.strftime('%d-%m-%Y %H:%M:%S'),
                                                                              (time.time() - tic)/60,
                                                                              train_score,
                                                                              test_score,
                                                                              space] + space_values
        return test_score
    
    def fmin(self, space, max_evals=500, prints=False):
        best = hpo.fmin(fn=self.objective,
                        space=space,
                        algo=hpo.tpe.suggest,
                        max_evals=max_evals,
                        trials=self.trials,
                        rstate=np.random.seed(self.seed),
                        verbose=True)
        
    def save(self):
        with open(os.path.join(self.models_path, f'best_model_{self.model_name}.pkl'), 'wb') as f:
            dill.dump(self, f)
            
            
            
            