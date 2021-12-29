"""Sets up environment for exploratory analyses"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def news_regression(y_col, x_col, df, lags=12):
    x_cols = []
    for i in range(lags+1):
        new_col = f'{x_col}-lag{i}'
        df[new_col] = df[f'{x_col}'].shift(i)
        x_cols.append(new_col)
    df = df.dropna()
    return sm.OLS(df[y_col], sm.add_constant(df[x_cols])).fit()