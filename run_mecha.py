from utils.mecha import *

ys = {'DJCI', 'BCI', 'CRBI', 'CRBI_industrials', 'CRBI_food',	'CRBI_metals', 'CRBI_livestock', 'CRBI_fats', 'CRBI_textiles',
'lumber', 'oats', 'palladium', 'rubber', 'gold','hog', 'wheat', 'crude', 'gas', 'GSCI', 'GSNECI', 'GSECI'}
xs = {'SPX', 'MSCI_W', 'MSCI_EM'}

ADAM(ys, xs).run(file_ending='-master')