from utils.mecha import *

ys = {'WTI', 'GSCI', 'GSECI', 'GSNECI', 'heating_oil', 'gas', 'propane'}
xs = ys.union({'SPX', 'MSCI_ACWI', 'MSCI_W', 'MSCI_EM'})

ADAM(ys, xs).run()