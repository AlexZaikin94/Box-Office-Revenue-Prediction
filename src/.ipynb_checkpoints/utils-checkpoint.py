import pandas as pd



def corr(X, y):
    _X = X.copy()
    _X['y'] = y
    corr_vals = _X.corr()['y'].sort_values(key=lambda x: abs(x), ascending=False)
    return pd.DataFrame(corr_vals.drop('y'))
