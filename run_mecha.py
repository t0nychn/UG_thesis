from utils.mecha import *

ys = {'gold', 'hog', 'lumber', 'oats', 'palladium', 'rubber', 'wheat'}
xs = {'SPX', 'MSCI_W', 'MSCI_EM', 'GSCI', 'GSNECI', 'GSECI'}

ADAM(ys, xs).run(file_ending='2')