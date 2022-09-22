from . import *
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statistics import NormalDist

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
    def __init__(self, x0=0, p=0, r=0, Q=0, lags=0):
        self.lags = lags
        self.p = p
        self.r = r
        self.Q = Q
        self.x0 = x0
    
    def run(self, y_col, x_col, df, show_scatter=False, smooth=False):
        """Latest version capable of smoothing (lag 0 only)"""
        if show_scatter:
            plt.scatter(df[x_col], df[y_col])
            plt.title('Input Data')
            plt.ylabel(y_col)
            plt.xlabel(x_col)
            plt.show()
        kf = KalmanFilter(dim_x=self.lags+1, dim_z=self.lags+1)
        if self.x0 == 0:
            kf.x = np.array([0 for _ in range(self.lags+1)])
        kf.F = np.diag(np.diag(([[1 for _ in range(self.lags+1)] for _ in range(self.lags+1)])))
        
        df = df.dropna()
        if self.p == 0:
            self.p = df[y_col].var()
        if self.r == 0:
            self.r = df[y_col].var()
        if self.Q == 0:
            kf.Q = 0.01
        kf.P = np.diag([self.p for i in range(self.lags+1)])
        kf.R = np.diag([self.r for i in range(self.lags+1)])

        zs = [np.array([df.iloc[i-l][y_col] for l in range(self.lags+1)]) for i in range(self.lags+1, len(df))]
        Rs = [kf.R for _ in range(self.lags+1, len(df))]
        Hs = [np.array([[df.shift(l).iloc[i-j][x_col] for l in range(self.lags+1)]
                                 for j in range(self.lags+1)]) for i in range(self.lags+1, len(df))]
        Fs = [kf.F for _ in range(self.lags+1, len(df))]
        Qs = [kf.Q for _ in range(self.lags+1, len(df))]

        mu, cov, _, _ = kf.batch_filter(zs, Fs=Fs, Rs=Rs, Qs=Qs, Hs=Hs)
        if smooth:
            mu, cov, _, _ = kf.rts_smoother(mu, cov, Fs=Fs, Qs=Qs)

        # unpack into dicts
        xs = {i: [] for i in range(self.lags+1)}
        ps = {i: [] for i in range(self.lags+1)}
        for i in range(len(mu)):
            for j in range(self.lags+1):
                xs[j].append(mu[i][j])
                ps[j].append(np.sqrt(cov[i].diagonal()[j]))

        # save for backtesting
        self.x_col = x_col
        self.df = df
        self.b_df = pd.DataFrame(xs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.lags+1, len(df))))
        self.p_df = pd.DataFrame(ps, index=self.b_df.index)
        return self.b_df.cumsum(axis=1)
    
    def _run_depr(self, y_col, x_col, df, show_scatter=False):
        """Initial implementation - deprecated by batch implementation which allows for smoothing"""
        if show_scatter:
            plt.scatter(df[x_col], df[y_col])
            plt.title('Input Data')
            plt.ylabel(y_col)
            plt.xlabel(x_col)
            plt.show()
        self.xs = {i: [] for i in range(self.lags+1)}
        ps = {i: [] for i in range(self.lags+1)}
        kf = KalmanFilter(dim_x=self.lags+1, dim_z=self.lags+1)
        if self.x0 == 0:
            kf.x = np.array([0 for _ in range(self.lags+1)])
        kf.F = np.diag(np.diag(([[1 for _ in range(self.lags+1)] for _ in range(self.lags+1)])))
        
        df = df.dropna()
        if self.p == 0:
            self.p = df[y_col].var()
        if self.r == 0:
            self.r = df[y_col].var()
        if self.Q == 0:
            kf.Q = 0.01
        kf.P = np.diag([self.p for i in range(self.lags+1)])
        kf.R = np.diag([self.r for i in range(self.lags+1)])
        for i in range(self.lags+1, len(df)):
            kf.predict()
            kf.H = np.array([[df.shift(l).iloc[i-j][x_col] for l in range(self.lags+1)]
                                 for j in range(self.lags+1)]) # update H with fresh values
            kf.update(np.array([df.iloc[i-l][y_col] for l in range(self.lags+1)]))
            for j in range(self.lags+1):
                self.xs[j].append(kf.x[j])
                ps[j].append(np.sqrt(kf.P.diagonal()[j])) # collect std         

        # save for backtesting
        self.x_col = x_col
        self.df = df
        self.b_df = pd.DataFrame(self.xs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.lags+1, len(df))))
        self.p_df = pd.DataFrame(ps, index=self.b_df.index)
        return self.b_df.cumsum(axis=1)
    
    def backtest(self):
        """Returns predicted dependent variables"""
        backtest = self.df.copy(deep=True)
        backtest['res'] = np.sum(self.b_df[i] * backtest[f'{self.x_col}'].shift(i) for i in range(self.lags+1))
        return backtest.dropna()['res']
    
    def shade_cred_intervals(self, p=0.1, alpha=0.08):
        crit = NormalDist(0, 1).inv_cdf(1 - p)
        self.c_df = pd.DataFrame({'lower': self.b_df.sum(axis=1) - crit * self.p_df.sum(axis=1), 'upper': self.b_df.sum(axis=1) + crit * self.p_df.sum(axis=1)}, index=self.b_df.index)
        plt.fill_between(self.c_df.index, self.c_df['lower'], self.c_df['upper'], alpha=alpha, color='purple')
    
    def calc_likelihood(self, val=0, direction='>', p=0.1):
        """Refactored from plot_likelihood to use in mecha"""
        probs = {i: [] for i in range(self.lags+1)}
        self.sigs = {} # save significant ones for highlighting
        for ind in self.b_df.index:
            for j in range(self.lags+1):
                dist = NormalDist(self.b_df.loc[ind][j], self.p_df.loc[ind][j])
                if direction == '>':
                    prob = 1 - dist.cdf(val)
                elif direction == '<':
                    prob = dist.cdf(val)
                else:
                    raise ValueError('Direction needs to be either > (default) or <')
                probs[j].append(prob)
                if prob >= 1-p:
                    self.sigs[ind] = prob
        return probs

    def plot_likelihood(self, val=0, direction='>', p=0.1, figsize=(20,3), recessions=True):
        probs = self.calc_likelihood(val=val, direction=direction, p=p)
        probs_df = pd.DataFrame(probs, index=self.b_df.index)
        probs_df[0].plot(figsize=figsize, title=f'P(state {direction} {val})', alpha=0)
        plt.axvline('2000-01-01', linestyle='--', color='black')
        plt.fill_between(probs_df.index, 0, probs_df[0], alpha=0.3, color='purple')
        if recessions:
            r = pd.read_csv('data/recessions.csv')
            for index, row in r.iterrows():
                plt.axvspan(pd.to_datetime(row['start'], dayfirst=True), pd.to_datetime(row['end'], dayfirst=True), color='grey', alpha=0.2)
        if p > 0:
            plt.scatter(self.sigs.keys(), self.sigs.values(), color='blue')


class KFConst:
    """Adds risk premia constant - doesn't work very well"""
    def __init__(self, x0=0, p=0, r=0, Q=0, lags=0):
        self.lags = lags
        self.p = p
        self.r = r
        self.Q = Q
        self.x0 = x0
    
    def run(self, y_col, x_col, df, show_scatter=False):
        if show_scatter:
            plt.scatter(df[x_col], df[y_col])
            plt.title('Input Data')
            plt.ylabel(y_col)
            plt.xlabel(x_col)
            plt.show()
        self.xs = {i: [] for i in range(self.lags+2)}
        kf = KalmanFilter(dim_x=self.lags+2, dim_z=self.lags+2)
        if self.x0 == 0:
            kf.x = np.array([0] + [1 for _ in range(self.lags+1)])
        kf.F = np.array([[1 for _ in range(self.lags+2)] for _ in range(self.lags+2)]) 
        
        df = df.dropna()
        if self.p == 0:
            self.p = df[y_col].var()
        if self.r == 0:
            self.r = df[y_col].var()
        if self.Q == 0:
            kf.Q = df[x_col].var()
        kf.P = np.diag([self.p for i in range(self.lags+2)])
        kf.R = np.diag([self.r for i in range(self.lags+2)])
        for i in range(self.lags+2, len(df)):
            kf.predict()
            kf.H = np.array([[1] + [df.shift(l).iloc[i-j][f'{x_col}'] for l in range(self.lags+1)]
                                 for j in range(self.lags+2)]) # update H with fresh values
            kf.update(np.array([df.iloc[i-l][f'{y_col}'] for l in range(self.lags+2)]))
            for j in range(self.lags+2):
                self.xs[j].append(kf.x[j])

        # save for backtesting
        self.x_col = x_col
        self.df = df
        self.b_df = pd.DataFrame(self.xs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.lags+2, len(df))))
        return self.b_df.cumsum(axis=1)
    
    def backtest(self):
        """Returns predicted dependent variables"""
        backtest = self.df.copy(deep=True)
        backtest['res'] = self.b_df[0] + np.sum(self.b_df[i] * backtest[f'{self.x_col}'].shift(i) for i in range(1, self.lags+2))
        return backtest.dropna()['res']

def ols_backtest(x, model, lags=0):
    backtest = x * model.params[0]
    for i in range(1, lags+1): # remember params contain constant at [0]
        backtest += x.shift(i-1) * model.params[i]
    return backtest

def plot_backtests(y, x_label, res_dict, rmse=True, start=0, figsize=(20,5), plot=False):
    """Better version is below. This function maintained for backwards compatibility"""
    if start == 0:
        start = min(y.index)
    fig, ax = plt.subplots(len(res_dict), sharex=True, figsize=figsize)
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

def backtest(y, res_dict, x_label=False, rmse=True, start=0, figsize=(20,5), plot=False):
    if start == 0:
        start = min(y.index)
    if plot:
        fig, ax = plt.subplots(len(res_dict), sharex=True, figsize=figsize)
    i = 0
    for k in res_dict.keys():
        if x_label:
            fig.suptitle(f'Backtesting Results for {x_label}')
        if plot:
            ax[i].plot(res_dict[k].loc[start:max(y.index)], label=f'{k}', alpha=0.8)
            ax[i].plot(y.loc[start:max(y.index)], label='actual', alpha=0.8)
            ax[i].set_title(f'{k}')
            ax[i].legend()
        if rmse:
            if i == 0:
                print(f'RMSE Random Walk: {np.sqrt(np.sum((y - 0).dropna() ** 2) / len(y))}')
            print(f'RMSE {k}: {np.sqrt(np.sum((y - res_dict[k]).dropna() ** 2) / len(res_dict[k]))}')
        i += 1

def hp_kalman_plot(df, hp_trend=True, hp_cycle=False, hp_fill=False, figsize=(20,5), title=False, linewidth=1.5, splitline=True, recessions=True, geopolitics=False, legend=True):
    df.plot(figsize=figsize, label='Kalman estimates', linewidth=linewidth, legend=legend)
    c, t = hpfilter(df.iloc[:,-1], lamb=129600)
    if hp_trend:
        t.plot(label='HP trend', color='b', alpha=0.8, linewidth=linewidth)
        if hp_fill:
            plt.fill_between(t.index, [0 for _ in t.index], t, alpha=0.5)
    if hp_cycle:
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
            plt.axvspan(pd.to_datetime(row['start'], dayfirst=True), pd.to_datetime(row['end'], dayfirst=True), alpha=0.15, color=row['colour'], label=row['name'])
    if legend:
        if geopolitics:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fancybox=True, shadow=True)
        else:
            plt.legend()
    plt.plot()