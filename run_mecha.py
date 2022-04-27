from utils.mecha import *

ys = {'DJCI', 'lumber', 'oats', 'palladium', 'rubber', 'gold','hog', 'wheat', 'crude', 'gas', 'GSCI', 'GSNECI', 'GSECI'}
xs = {'SPX', 'MSCI_W', 'MSCI_EM'}

ADAM(ys, xs).run(file_ending='-master')