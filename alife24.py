
from gol_info_utils import *
# import pandas as pd

'''
alife24 more complete experiments
'''

# load full blinker codomain
# bk_dxys = load_data(filename=f'gol_tx_domains_cap=no_cap=20_blinker')
# print_ac_cases(bk_dxys)                                         # shape = (1,048,576, 9, 7)
# remove < 3 cases
# bk_dxys = sum_higher(bk_dxys,3,arrays=True)                     
# print_ac_cases(bk_dxys)                                         # 1,047,726
# c3: 5901, c4: 20174, c5: 42240; c3-c5: 68315

# filter and reduce codomain
# from gol_fxs import *
# bk_dxys = apply_ct(bk_dxys,blinker.dx,rm_zeros=True)            # 1,047,726 (same)
# bk_dxys = rm_non_sx_cells(bk_dxys,blinker.dx)                   # 1,009,842; 3=55541, 4=57228, 5=60838
# bk_dxys = mk_yz_decay(bk_dxys,blinker.dx,expanded=False)        # 965436; 3=15175, 4=53410; 5=60620
# bk_dxys = sum_in_range(bk_dxys,3,6,arrays=True)                 # 129,205
# print_ac_cases(bk_dxys)
# save_as(bk_dxys,name='alife2024_blinker_dxys_cap=5',ext='gol')

# search symsets
# bk_dxys = load_data('alife2024_blinker_dxys_cap=5.gol',dirname='')
# print_ac_cases(bk_dxys)
# look for patterns with 5 or less active cells
# syms,sym_cases,sym_ids = mk_symsets_large_dxs(bk_dxys,membrane=True,search_borders=True,ids=True)
# => symsets=71: 3acs=2, 4acs=15, 5acs=54 (same with and without searching border)
# save_as([syms,sym_cases,sym_ids],name='bk_syms_full_cap=5',ext='gol')
# syms,sym_cases,sym_ids = load_data('bk_syms_full_cap=5.gol',dirname='')
# print_ac_cases(syms)

# patterns for alife24
pxs = [blinker,pb0,block,ttz,ttt,ttl,bar,baby,flag,kite,worm,gla,glb,zz]

from gol_fxs import *

# for every pattern in pxs
for px in pxs:
    print(f'\n\n{px.label}\n\n')
    # load codomains data
    dxys = load_data(filename=f'gol_tx_domains_cap=no_cap={px.env.sum()}_{px.label}',ext='gol_info')
    print(f'total cases = {dxys.shape[0]}')
    # print_ac_cases(dxys)
    # for long (ac>22), heavy cases
    if dxys.shape[0] > 2**22:
        dxys = sum_lower(dxys,13,arrays=True)
    # disregard cases < 3
    dxys = sum_higher(dxys,3,arrays=True)
    print_ac_cases(dxys)
    # filtering
    dxys = rm_non_sx_cells(dxys,px.dx)
    dxys = mk_yz_decay(dxys,px.dx,expanded=False)
    # only domains <= 5
    dxys = sum_in_range(dxys,3,6,arrays=True)
    print_ac_cases(dxys)
    save_as(dxys,name=f'alife24_{px.label}_dxys_cap=5',fdir='gol_exps_data',ext='gol')
    # symset search
    syms,sym_cases,sym_ids = mk_symsets_large_dxs(dxys,membrane=True,search_borders=False,ids=True)
    save_as([syms,sym_cases,sym_ids],name=f'alife24_{px.label}_symsets_full_cap=5',fdir='gol_exps_data',ext='gol')
    print_ac_cases(syms)