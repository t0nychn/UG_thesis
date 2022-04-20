"""This file gives a general framework to automate analysis given input CSV and saves
results to MySQL database"""

from models import *

class ADAM:
    """Automated Deployment of Analytical Methods"""
    def __init__(self, y_series, x_series, df, lags=0):
        """Runs analyses pairing every y with every x"""
        self.ys = y_series
        self.xs = x_series
        self.input_df = df
        self.output_df = pd.DataFrame(index=df.index[(lags+1):])
        self.kf = KF(lags=lags)
        self.lags=lags

    def run(self, save_path='data/saved/ADAM.csv'):
        ycount = 0
        xcount = 0

        for y in self.ys:
            ycount += 1
            for x in self.xs:
                xcount += 1

                betas = self.kf.run(y, x, self.input_df)
                for i in range(self.lags+1):
                    self.output_df[f'{y}-{x}_betas{i}'] = betas[i]
                self.output_df[f'{y}-{x}_probs'] = self.kf.calc_likelihood()
                self.output_df[f'{y}-{x}_backtest'] = [self.kf.backtest] + [0 for _ in range(len(self.output_df))]

                if self.lags==0:
                    self.output_df[f'{y}-{x}_betas_smth{i}'] = self.kf.run(y, x, self.input_df, smooth=True)
                    self.output_df[f'{y}-{x}_probs_smth'] = self.kf.calc_likelihood()
                    self.output_df[f'{y}-{x}_backtest_smth'] = [self.kf.backtest] + [0 for _ in range(len(self.output_df))]

                self.output_df.to_csv(save_path)

                print(f'Finished analysing {y} and {x}! Results saved to CSV.')
                print(f'{ycount}/{len(self.ys)} of ys completed and {xcount}/{len(self.xs)} of xs completed')
                
