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
    return pd.read_csv(filepath, parse_dates=True, index_col=index, dayfirst=True)

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
    return pd.DataFrame(series.rename(col)).dropna()

def calc_shock(col, df, method='mad2', clean=True, standardize=True, seasonal=True):
    """Calculates shock from index value. Can use either naive method in Exploratory1 or 
    moving average method in Exploratory2.
    Returns input column as new DataFrame."""
    series = df.copy(deep=True)[col]
    if method[:3] == 'mad':
        ma = series.rolling(int(method[-1])).mean()
        series = (series - ma).diff()
        if clean:
            return clean_series(col, pd.DataFrame(series.rename(col)).dropna(), standardize=standardize, seasonal=seasonal)
        else:
            return pd.DataFrame(series.rename(col)).dropna()
    elif method == 'naive':
        series = series.pct_change().diff()
        if clean:
            return clean_series(col, pd.DataFrame(series.rename(col)).dropna(), standardize=standardize, seasonal=seasonal)
        else:
            return pd.DataFrame(series.rename(col)).dropna()
    else:
        raise ValueError('Check correct method specified!')


# data visualization
def draw(models, start=2, periods=12, conf_int=True, legend=True, cumulative=False, figsize=(6.4,4.8), labels=None, colors=None, alpha=0.1):
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
        if conf_int:
            if cumulative:
                confs = models[i].conf_int()
                if colors != None:
                    plt.fill_between([*range(periods+1)], np.cumsum(confs[0][start:]), np.cumsum(confs[1][start:]), alpha=alpha, color=colors[i])
                else:
                    plt.fill_between([*range(periods+1)], np.cumsum(confs[0][start:]), np.cumsum(confs[1][start:]), alpha=alpha)
            else:
                confs = models[i].conf_int()
                if colors != None:
                    plt.fill_between([*range(periods+1)], confs[0][start:], confs[1][start:], alpha=alpha, color=colors[i])
                else:
                    plt.fill_between([*range(periods+1)], confs[0][start:], confs[1][start:], alpha=alpha)
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

def ardl(y_col, x_col, df, lags=[1, 12]):
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