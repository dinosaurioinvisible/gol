# experiments with membranes

from gol_fxs import *

# blinker to blinker count
bk = mk_gol_pattern('blinker')[0]
bk_dx = expand_domain(bk)

def blinker_to_blinker():
    bk = mk_gol_pattern('blinker')[0]
    bk_dx = expand_domain(bk)
    bk_dxs = mk_env_binary_domains(bk,membrane=True)
    bk_dxs,bk_dxys = vl_multi_gol_step(bk_dxs,bk_dx,rm_zeros=True)
    bk_dxys = check_act_ct(bk_dxys,bk_dx,ids=False)
    bks = is_sx_in_dxs(bk,bk_dxys,bk_dx,membrane=True,moore_nb=False)
    return bks
# bks = blinker_to_blinker()

# what a blinker produces
def fwd_fx(load_pf=True,load_ft=True):
    bk = mk_gol_pattern('blinker')[0]
    bk_dx = expand_domain(bk)
    if load_pf:
        try:
            bk_dxs,bk_dxys = load_data('bk_pf.gol')
        except:
            load_pf = False
    if not load_pf:
        bk_dxs = mk_env_binary_domains(bk,membrane=True)
        # -> 2^20 = 1,048,576 envs
        bk_dxs,bk_dxys = vl_multi_gol_step(bk_dxs,bk_dx,rm_zeros=True)
        # -> 1,047,612
        bk_dxys,ct_ids = check_act_ct(bk_dxys,bk_dx,ids=True)
        bk_dxs = bk_dxs[ct_ids]
        # -> 1,047,612 (same)
        save_as([bk_dxs,bk_dxys],'bk_pf.gol')

    if load_ft:
        try:
            bx5,by5 = load_data('bk_xy.gol')
        except:
            load_ft = False
    if not load_ft:
        # basic patterns
        b5_ids = sum_lower(bk_dxys,6)
        bx5 = bk_dxs[b5_ids]
        by5 = bk_dxys[b5_ids]
        print_ac_cases(by5)
        # 98,717 (3:7533, 4:27626, 5:63558)
        # save_as([bx5,by5],'bk_xy.gol')
        
        # filtering
        # remove 1-env cells
        by5,ids = rm_env_cells_dv1(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 67,455: 3:25622, 4:27494, 5:14339
        # remove 2-env cells 
        by5,ids = rm_env_cells_dv2(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 62,057: 3:28428, 4:22096, 5:11533
        # yz decay
        by5,ids = mk_yz_decay(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 30,965: (3:3278, 4:17562, 5:10125)
        # rm non env cells
        by5,ids = rm_non_env(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 30,965: (3:3770, 4:18620, 5:8575)
        # 2ble check 
        # (because of yz decay and removing)
        by5,ids = rm_env_cells_dv1(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 30,965: (3:4774, 4:17616, 5:8575)
        by5,ids = rm_env_cells_dv2(by5,bk_dx,ids=True)
        bx5 = bx5[ids]
        # 30,965: (3:4774, 4:17616, 5:8575)
        save_as([bx5,by5],'bk_xy.gol')

    # symsets
    bk_syms,sym_cases,sym_ids = mk_symsets_large_dxs(by5,bk_dx,membrane=True,ids=True)
    # 94 => 3:2, 4:20, 5:72
    # 77 => 3:2, 4:20, 5:55
    # 67 => 3:2, 4:15, 5:48
    save_as([bk_syms,sym_cases,sym_ids],'bk_syms.gol')
    # reduce env symsets 

# fwd_fx()
# bx,by = load_data('bk_pf.gol')
# bx5,by5 = load_data('bk_xy.gol')
# bk_syms,cases,ids = load_data('bk_syms.gol')

