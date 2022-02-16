from . import *
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.filters.hp_filter import hpfilter

class DL:
    def __init__(self, y_col, x_col, df, lags=12, const=False):
        self.model = dl(y_col, x_col, df, lags=lags, const=const).get_robustcov_results(cov_type='HAC', maxlags=1) # HAC due to resids being AR(1)
        self.lags = lags
    
    def summary(self):
        return self.model.summary()
    
    def plot(self, figsize=(6.4,4.8)):
        coeffs = self.model.params[1:]
        bse = self.model.bse[1:]
        plt.figure(figsize=figsize)
        plt.plot(np.cumsum(coeffs))
        bands = pd.DataFrame({0: [coeffs[x] - bse[x] for x in range(len(coeffs))], 1: [coeffs[x] + bse[x] for x in range(len(coeffs))]})
        plt.fill_between([*range(self.lags+1)], np.cumsum(bands[0]), np.cumsum(bands[1]), alpha=0.1)
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.show()


class ARDL:
    """Slightly harder than DL class because we have to simulate results rather than taking response from ceoffs"""
    def __init__(self, y_col, x_col, df, lags=[1, 11]):
        self.model = ardl(y_col, x_col, df, lags=lags).get_robustcov_results()
        self.lags = lags
    
    def summary(self):
        return self.model.summary()

    def simulate_response(self):
        ar_coeffs = self.model.params[1:self.lags[0]+1] # selects autoregressive coeffs (avoiding constant)
        dl_coeffs = self.model.params[self.lags[0]+1:] # selects non-autoregressive coeffs
        responses = []
        for i in range(self.lags[0] + self.lags[1] + 1):
            ar_component = 0
            for j in range(len(ar_coeffs)):
                if j < len(responses):
                    ar_component += ar_coeffs[j] * responses[-(j+1)]
                else:
                    break
            dl_component = sum(dl_coeffs[:min(i+1, len(dl_coeffs))])
            responses.append(ar_component + dl_component)
        return responses

    def simulate_bse(self):
        ar_coeffs = self.model.params[1:self.lags[0]+1] # selects autoregressive coeffs (avoiding constant)
        dl_bse = self.model.bse[self.lags[0]+1:]
        errors = []
        for i in range(self.lags[0] + self.lags[1] + 1):
            ar_component = 0
            for j in range(len(ar_coeffs)):
                if j < len(errors):
                    ar_component += ar_coeffs[j] * errors[-(j+1)]
                else:
                    break
            dl_component = sum(dl_bse[:min(i+1, len(dl_bse))])
            errors.append(ar_component + dl_component)
        return errors

    def plot(self, figsize=(6.4,4.8), *args):
        responses = self.simulate_response()
        errors = self.simulate_bse()
        bands = pd.DataFrame({0: [responses[x] - errors[x] for x in range(len(responses))], 1: [responses[x] + errors[x] for x in range(len(responses))]})
        plt.figure(figsize=figsize)
        plt.plot(responses, *args)
        plt.fill_between([*range(self.lags[0] + self.lags[1] + 1)], bands[0], bands[1], alpha=0.1)
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.show()


class KF:
    def __init__(self, x0, p=0, r=0, Q=0, lags=0):
        self.lags = lags
        kf = KalmanFilter(dim_x=lags+1, dim_z=lags+1)
        kf.x = x0
        kf.F = np.array([[1 for _ in range(self.lags+1)] for _ in range(self.lags+1)])
        self.kf = kf
        self.xs = {i: [] for i in range(self.lags+1)}
        self.p = p
        self.r = r
        self.Q = Q
    
    def run(self, y_col, x_col, df):   
        df = df.dropna()
        if self.p == 0:
            self.p = df[f'{x_col}'].var()
        if self.r == 0:
            self.r = df[f'{x_col}'].var()
        if self.Q == 0:
            self.kf.Q = df[f'{y_col}'].var()
        self.kf.P = np.diag([self.p for i in range(self.lags+1)])
        self.kf.R = np.diag([self.r for i in range(self.lags+1)])
        for i in range(self.lags+1, len(df)):
            self.kf.predict()
            self.kf.H = np.array([[df.shift(l).iloc[i-j][f'{x_col}'] for l in range(self.lags+1)]
                                 for j in range(self.lags+1)]) # update H with fresh values
            self.kf.update(np.array([df.iloc[i-l][f'{y_col}'] for l in range(self.lags+1)]))
            for j in range(self.lags+1):
                self.xs[j].append(self.kf.x[j])

        # save for backtesting
        self.x_col = x_col
        self.df = df
        self.b_df = pd.DataFrame(self.xs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.lags+1, len(df))))
        return self.b_df.cumsum(axis=1)
    
    def backtest(self):
        """Returns predicted dependent variables"""
        backtest = self.df.copy(deep=True)
        backtest['res'] = np.sum(self.b_df[i] * backtest[f'{self.x_col}'].shift(i) for i in range(self.lags+1))
        return backtest.dropna()['res']

def ols_backtest(x, model, lags=0):
    backtest = x * model.params[0]
    for i in range(1, lags+1): # remember params contain constant at [0]
        backtest += x.shift(i-1) * model.params[i]
    return backtest

def plot_backtests(y, x_label, res_dict, rmse=True, start=0):
    if start == 0:
        start = min(y.index)
    fig, ax = plt.subplots(len(res_dict), sharex=True, figsize=(20,8))
    fig.suptitle(f'Backtesting Results for {x_label}')
    i = 0
    for k in res_dict.keys():
        ax[i].plot(res_dict[k].loc[start:max(y.index)], label=f'{k}', alpha=0.8)
        ax[i].plot(y.loc[start:max(y.index)], label='actual', alpha=0.8)
        ax[i].set_title(f'{k}')
        ax[i].legend()
        if rmse:
            if i == 0:
                print(f'RMSE Random Walk: {np.sqrt(np.sum((y - 0).dropna() ** 2) / len(y))}')
            print(f'RMSE {k}: {np.sqrt(np.sum((y - res_dict[k]).dropna() ** 2) / len(res_dict[k]))}')
        i += 1

def hp_kalman_plot(df, cycle=False, figsize=(20,7), title=False, linewidth=1.5, splitline=True, recessions=True, geopolitics=False, legend=True):
    df.plot(figsize=figsize, label='Kalman estimates', linewidth=linewidth)
    c, t = hpfilter(df.iloc[:,-1], lamb=129600)
    t.plot(label='HP trend', color='b', alpha=0.8, linewidth=linewidth)
    plt.fill_between(t.index, [0 for _ in t.index], t, alpha=0.5)
    if cycle:
        c.plot(label='HP cycle')
    plt.axhline(0, linestyle='--', color='grey', linewidth=linewidth)
    if splitline:
        plt.axvline('2000-01-01', linestyle='--', color='black', linewidth=linewidth)
    if title:
        plt.title(title)
    if recessions:
        r = pd.read_csv('data/recessions.csv')
        for index, row in r.iterrows():
            plt.axvspan(pd.to_datetime(row['start'], dayfirst=True), pd.to_datetime(row['end'], dayfirst=True), color='grey', alpha=0.2)
    if geopolitics:
        g = pd.read_csv('data/GPR_events.csv')
        for index, row in g.iterrows():
            plt.axvspan(pd.to_datetime(row['start'], dayfirst=True), pd.to_datetime(row['end'], dayfirst=True), alpha=0.2, color=row['colour'], label=row['name'])
    if legend:
        plt.legend()
    plt.plot()
