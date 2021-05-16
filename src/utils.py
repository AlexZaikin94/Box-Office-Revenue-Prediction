import pandas as pd



def corr(X, y, only_y=True, drop_y_index=True):
    _X = X.copy()
    _X['y'] = y
    corr_vals = _X.corr().sort_values('y', key=lambda x: abs(x), ascending=False)
    
    if drop_y_index:
        corr_vals = corr_vals.drop(index='y')
    
    if only_y:
        corr_vals = corr_vals[['y']]
    else:
        corr_vals = corr_vals[list(corr_vals.index)]
    return corr_vals
