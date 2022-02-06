from . import *

class DL:
    def __init__(self, y_col, x_col, df, lags=12):
        self.model = dl(y_col, x_col, df, lags=lags).get_robustcov_results(cov_type='HAC', maxlags=1) # HAC due to resids being AR(1)
        self.lags = lags
    
    def summary(self):
        return self.model.summary()
    
    def plot(self, figsize=(6.4,4.8)):
        params = self.model.params[1:]
        bse = self.model.bse[1:]
        plt.figure(figsize=figsize)
        plt.plot(np.cumsum(params))
        bands = pd.DataFrame({0: [params[x] - bse[x] for x in range(len(params))], 1: [params[x] + bse[x] for x in range(len(params))]})
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