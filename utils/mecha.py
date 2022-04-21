"""This file gives a general framework to automate analysis given input CSV and saves
results to MySQL database"""

from pydantic import MissingError
from .models import *

class ADAM:
    """Automated Deployment of Analytical Methods"""
    def __init__(self, y_series, x_series, df=0, lags=0):
        """Runs analyses pairing every y with every x"""
        self.ys = y_series
        self.xs = x_series
        if df == 0:
            self.input_df = load('data/master.csv').pct_change()
        else:
            self.input_df = df.pct_change()

        # create empty dataframes for three output types
        self.output_betas = pd.DataFrame(index=self.input_df.index[(lags+2):])
        self.output_probs = pd.DataFrame(index=self.input_df.index[(lags+2):])
        self.output_backtests = pd.DataFrame(index=self.input_df.index[(lags+2):])
        self.output_len = len(self.output_betas)

        # initiate Kalman
        self.kf = KF(lags=lags)
        self.lags=lags

    def run(self, save_path='data/saved/'):
        ycount = 0
        xcount = 0

        for y in self.ys:
            ycount += 1
            clean_y = clean_series(y, self.input_df)

            for x in self.xs:
                xcount += 1
                clean_x = clean_series(x, self.input_df)
                df = clean_y.join(clean_x)

                betas = self.kf.run(y, x, df)
                missing = self.output_len - len(betas)
                for i in range(self.lags+1):
                    self.output_betas[f'{y}-{x}-{i}'] = [*[np.nan for _ in range(missing)] + list(betas[i])]
                self.output_probs[f'{y}-{x}'] = [*[np.nan for _ in range(missing)] + list(self.kf.calc_likelihood()[0])]
                self.output_backtests[f'{y}-{x}'] = self.kf.backtest()

                if self.lags==0:
                    self.output_betas[f'{y}-{x}-smth'] = [*[np.nan for _ in range(missing)] + list(self.kf.run(y, x, df, smooth=True)[0])]
                    self.output_probs[f'{y}-{x}-smth'] = [*[np.nan for _ in range(missing)] + list(self.kf.calc_likelihood()[0])]
                    self.output_backtests[f'{y}-{x}-smth'] = self.kf.backtest()

                self.output_betas.to_csv(save_path + 'betas.csv')
                self.output_probs.to_csv(save_path + 'probs.csv')
                self.output_backtests.to_csv(save_path + 'backtests.csv')

                print(f'Finished analysing {y} and {x}! Results saved to CSV.')
                print(f'{ycount}/{len(self.ys)} of ys completed and {xcount}/{len(self.xs)} of xs completed')

