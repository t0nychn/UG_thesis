"""Sets up environment for exploratory analyses"""
import sys
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt


# general
def env():
    print('--------- Dependencies ---------')
    print(f'python: {sys.version}')
    print(f'numpy: {np.__version__}')
    print(f'pandas: {pd.__version__}')
    print(f'matplotlib: {matplotlib.__version__}')
    print(f'statsmodels: {statsmodels.__version__}')


# data manipulation
def load(filepath, index='date'):
    """Shorthand for standardized import"""
    return pd.read_csv(filepath, parse_dates=True, index_col=index, dayfirst=True).sort_index()

def clean_series(col, df, standardize=True, seasonal=True):
    """standardize=True -> removes effect of changing variance by dividing values by rolling annual standard deviation. 
    seasonal=True -> removes effect of seasonality by subtracting values from collective average of each month across series.
    Returns input column as new DataFrame."""
    if standardize:
        series = df.copy(deep=True)[col]
        ann_vol = series.index.map(lambda x: series.groupby(series.index.year).std().loc[x.year])
        series = series/ann_vol
    if seasonal:
        mth_avg = series.index.map(lambda x: series.groupby(series.index.month).mean().loc[x.month])
        series = series - mth_avg
    return pd.DataFrame(series.rename(col)).dropna().sort_index()

def calc_shock(col, df, method='mad-12', clean=True, standardize=True, seasonal=True):
    """Calculates shock from index value. Can use either naive method in Exploratory1 or 
    moving average method in Exploratory2.
    Returns input column as new DataFrame."""
    series = df.copy(deep=True)[col]
    if method[:3] == 'mad':
        ma = series.rolling(int(method.split('-')[-1])).mean()
        series = series.pct_change() - ma.pct_change()
        if clean:
            return clean_series(col, pd.DataFrame(series.rename(col)).dropna(), standardize=standardize, seasonal=seasonal)
        else:
            return pd.DataFrame(series.rename(col)).dropna().sort_index()
    elif method == 'naive':
        series = series.pct_change().diff()
        if clean:
            return clean_series(col, pd.DataFrame(series.rename(col)).dropna(), standardize=standardize, seasonal=seasonal)
        else:
            return pd.DataFrame(series.rename(col)).dropna().sort_index()
    else:
        raise ValueError('Check correct method specified!')


# data visualization
def draw(models, start=1, periods=12, conf_int=False, bse=True, legend=True, cumulative=False, figsize=(6.4,4.8), labels=None, colors=None, alpha=0.1):
    plt.figure(figsize=figsize)
    for i in range(len(models)):
        model = models[i]
        if cumulative:
            if labels != None:
                plt.plot(np.cumsum(model.params[start:].reset_index(drop=True)), label=(labels[i] if labels != None else None), color=(colors[i] if colors != None else None))
            else:
                plt.plot(np.cumsum(model.params[start:].reset_index(drop=True)), label=f'series{i+1}')
        else:
            if labels != None:
                plt.plot(model.params[start:].reset_index(drop=True), label=(labels[i] if labels != None else None), color=(colors[i] if colors != None else None))
            else:
                plt.plot(model.params[start:].reset_index(drop=True), label=f'series{i+1}')
        if conf_int or bse:
            if bse:
                confs = pd.DataFrame({0: [model.params[x] - model.bse[x] for x in range(start, len(model.params))], 1: [model.params[x] + model.bse[x] for x in range(start, len(model.params))]})
            elif conf_int:
                confs = model.conf_int()[start:]
            if cumulative:
                if colors != None:
                    plt.fill_between([*range(periods+1)], np.cumsum(confs[0]), np.cumsum(confs[1]), alpha=alpha, color=colors[i])
                else:
                    plt.fill_between([*range(periods+1)], np.cumsum(confs[0]), np.cumsum(confs[1]), alpha=alpha)
            else:
                if colors != None:
                    plt.fill_between([*range(periods+1)], confs[0], confs[1], alpha=alpha, color=colors[i])
                else:
                    plt.fill_between([*range(periods+1)], confs[0], confs[1], alpha=alpha)
    plt.axhline(y=0, color='grey', linestyle='--')
    if legend:
        plt.legend()


# regressions
def dl(y_col, x_col, df, lags=12):
    """Regresses y against lagged values of x"""
    df_copy = df.copy(deep=True)
    x_cols = []
    for i in range(lags+1):
        new_col = f'{x_col}-lag{i}'
        df_copy[new_col] = df_copy[f'{x_col}'].shift(i)
        x_cols.append(new_col)
    df_copy = df_copy.dropna()
    return sm.OLS(df_copy[y_col], sm.add_constant(df_copy[x_cols])).fit()

def ardl(y_col, x_col, df, lags=[1, 6]):
    """Regresses y against lagged values of itself and x"""
    df_copy = df.copy(deep=True)
    x_cols = []
    for i in range(1, lags[0]+1):
        new_col = f'{y_col}-lag{i}'
        df_copy[new_col] = df_copy[f'{y_col}'].shift(i)
        x_cols.append(new_col)
    for i in range(lags[1]+1):
        new_col = f'{x_col}-lag{i}'
        df_copy[new_col] = df_copy[f'{x_col}'].shift(i)
        x_cols.append(new_col)
    df_copy = df_copy.dropna()
    return sm.OLS(df_copy[y_col], sm.add_constant(df_copy[x_cols])).fit()

def time_shift(y_col, x_col, df, start_yr=2000, end_yr=2020, direction='robust',model=dl, periods=12, start=1):
    """See Exploratory4"""
    coeffs_f = {i: [] for i in range(periods+1)}
    coeffs_b = {i: [] for i in range(periods+1)}
    coeffs_robust = {i: [] for i in range(periods+1)}

    for i in range(end_yr - start_yr):
        year = start_yr + i
        model_f = model(y_col, x_col, df[f'{year}':])
        model_b = model(y_col, x_col, df[:f'{year}'])
        params_f = np.cumsum(model_f.params[start:])
        params_b = np.cumsum(model_b.params[start:])
        for i in range(periods+1):
            coeffs_f[i].append(params_f[i])
            coeffs_b[i].append(params_b[i])
            coeffs_robust[i] = np.array(coeffs_f[i]) + np.array(coeffs_b[i])
    
    if direction == 'r':
        return coeffs_robust
    elif direction == 'f':
        return coeffs_f
    elif direction == 'b':
        return coeffs_b
    else:
        raise ValueError("Specify direction: 'f' (slices forward in time), 'b' (slices back in time), or 'r' (directionally robust)")