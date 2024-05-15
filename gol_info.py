from gol_info_utils import *
from gol_fxs import *



pb0_sx,pb0_dx = mk_gol_pattern('pb0',domain=True)
pb0_xs,pb0_ys,pb0_syms,pb0_cases,pb0_ids = load_data('xgol_pb0_cap=7.xgol')
pb0_uf = mk_uniform_ids(pb0.sx,'pt-block',pb0_syms,pb0_cases)

# sx & sx domain
# dxs and dxys (all)
# dxs and dxys (caps)
# sxys syms & sym cases 
# 

import pandas as pd


