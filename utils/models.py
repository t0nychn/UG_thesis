from . import *
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import t

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
        bands = pd.DataFrame({0: [coeffs[x] - bse[x] for x in range(len(coeffs))], 1: [bse[x] + bse[x] for x in range(len(coeffs))]})
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
    def __init__(self, window=36, lags=12, conf_int=0):
        """conf_int > 0 returns confidence intervals cumulated till last lag"""
        self.lags = lags
        self.window = window # number of months for measurement sample (+1 for indexing)
        self.conf_int = conf_int
    
    def run(self, y_col, x_col, df):
        betas = {i: [] for i in range(self.lags+1)}
        errors = []
        for i in range(len(df)-self.window-1):
            model = DL(y_col, x_col, df.iloc[i:self.window+i+1], lags=self.lags).model
            coeffs = model.params[1:]
            errors.append(np.sum(model.bse[1:]))
            for j in range(self.lags+1):
                betas[j].append(coeffs[j])
        b_df = pd.DataFrame(betas, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.window+1, len(df))))
        for i in range(self.lags+1):
            phi = AutoReg(np.array(b_df[i]), 1, trend='n').fit().params[-1]
            b_df[i] = b_df[i] * phi
        if self.conf_int > 0:
            confs = {}
            confs['lower'] = b_df.sum(axis=1) - np.array(errors) * t.ppf(1-self.conf_int/2, self.window-2*self.lags-2) # remember 2-tail test, df taken from downloaded chapter
            confs['upper'] = b_df.sum(axis=1) + np.array(errors) * t.ppf(1-self.conf_int/2, self.window-2*self.lags-2)
            return b_df.cumsum(axis=1), pd.DataFrame(confs, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.window+1, len(df))))
        else:
            return b_df.cumsum(axis=1)
    
    def run_ardl(self, y_col, x_col, df):
        betas = {i: [] for i in range(self.lags+1)}
        for i in range(len(df)-self.window-1):
            coeffs = ARDL(y_col, x_col, df.iloc[i:self.window+i]).simulate_response()
            for j in range(self.lags+1):
                betas[j].append(coeffs[j])
        b_df = pd.DataFrame(betas, index=(min(df.index) + pd.DateOffset(months=i) for i in range(self.window+1, len(df))))
        for i in range(self.lags+1):
            phi = AutoReg(np.array(b_df[i]), 1, trend='n').fit().params[-1]
            b_df[i] = b_df[i] * phi
        return b_df