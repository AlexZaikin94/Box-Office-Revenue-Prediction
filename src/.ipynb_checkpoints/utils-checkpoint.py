import numpy as np
import pandas as pd

from sklearn import metrics#, mean_squared_error, mean_squared_log_error



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