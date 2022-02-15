from . import *
from filterpy.kalman import KalmanFilter

class DL:
    def __init__(self, y_col, x_col, df, lags=12):
        self.model = dl(y_col, x_col, df, lags=lags).get_robustcov_results(cov_type='HAC', maxlags=1) # HAC due to resids being AR(1)
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
    def __init__(self, x0, p=100, r=10, Q=100, lags=0):
        self.lags = lags
        kf = KalmanFilter(dim_x=lags+2, dim_z=lags+2)
        kf.x = x0
        kf.Q = Q
        kf.P = np.diag([.5] + [p for i in range(self.lags+1)])
        kf.R = np.diag([.5] + [r for i in range(self.lags+1)])
        kf.F = np.array([[1 for _ in range(self.lags+2)] for _ in range(self.lags+2)])
        self.kf = kf
        self.xs = {i: [] for i in range(self.lags+2)}
    
    def run(self, y_col, x_col, df):   
        df = df.dropna()
        for i in range(self.lags+2, len(df)):
            self.kf.predict()
            self.kf.H = np.array([[1] + [df.shift(l).iloc[i-j][f'{x_col}'] for l in range(self.lags+1)]
                                 for j in range(self.lags+2)]) # update H with fresh values
            self.kf.update(np.array([df.iloc[i-l][f'{y_col}'] for l in range(self.lags+2)]))
            for j in range(self.lags+2):
                self.xs[j].append(self.kf.x[j])

        # save for backtesting
        self.x_col = x_col
        self.df = df
        self.b_df = pd.DataFrame(self.xs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.lags+2, len(df))))
        return self.b_df[[i for i in range(self.lags+1)]].cumsum(axis=1)
    
    def backtest(self):
        """Returns predicted dependent variables"""
        backtest = self.df.copy(deep=True).shift(self.lags)
        backtest['res'] = self.b_df[0] + np.sum(self.b_df[i+1] * backtest[f'{self.x_col}'].shift(i) for i in range(self.lags+1))
        return backtest.dropna()['res']

def ols_backtest(x, model, lags=0):
    backtest = model.params[0] + x * model.params[1]
    for i in range(2, lags+2): # remember params contain constant at [0]
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
        i += 1
        if rmse:
            print(f'RMSE {k}: {np.sqrt(np.sum((y - res_dict[k]).dropna()** 2) / len(res_dict[k]))}')