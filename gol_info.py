from gol_info_utils import *
import pandas as pd


'''
agency

1) mk all domains (dxs)
2) mk all codomains (dxys)
3) find py patterns in px codomains (structural transitions)
4) build transition map
5) examine tx map, output tx table
'''

# 1)
# mk_px_domains(pb0, cap=False, save=True)
# 2)
# mk_px_codomains(pb0, cap=False, save=True)
# 3)
# mk_px_data(pxs)
# 4) 
# mk_tx_map(pxs)
# 5) 
txmap = load_data(filename='gol_txmap_no_cap')
txs = get_tx_counts(txmap)


