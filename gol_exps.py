# experiments with membranes

from gol_fxs import *

# what a blinker produces
bk = mk_gol_pattern('blinker')[0]
bk_dx = expand_domain(bk)

def fwd_fx():
    bk_dxs = mk_env_binary_domains(bk,membrane=True)
    # -> 2^20 = 1,048,576 envs
    bk_dxs,bk_dxys = vl_multi_gol_step(bk_dxs,bk_dx,rm_zeros=True)
    # -> 1,047,612
    bk_dxys,ct_ids = check_act_ct(bk_dxys,bk_dx,ids=True)
    bk_dxs = bk_dxs[ct_ids]
    # -> 1,047,612 (same)
    # save_as([bk_dxs,bk_dxys],'bk_xys.gol')

    # basic patterns
    b5_ids = sum_lower(bk_dxys,6)
    bx5 = bk_dxs[b5_ids]
    by5 = bk_dxys[b5_ids]
    print_ac_cases(by5)
    # 98,717 (3:7533, 4:27626, 5:63558)
    # save_as([bx5,by5],'bk_xy.gol')

    # filtering
    by5,ids = rm_env_cells_dv12(by5,bk_dx,ids=True)
    bx5 = bx5[ids]
    # 64,747 (3:26956, 4:24786, 5:13005)
    by5,ids = mk_yz_decay(by5,bk_dx,ids=True)
    bx5 = bx5[ids]
    # 32,157 (3:3214, 4:17562, 5:11381)
    by5,ids = rm_non_env(by5,bk_dx,ids=True)
    bx5 = bx5[ids]
    # 32,157 (3:3710, 4:18620, 5:9827)
    # save_as([bx5,by5],'bk_xy.gol')

    # symsets
    bk_syms,sym_cases,sym_ids = mk_symsets_large_dxs(by5,bk_dx,membrane=True,ids=True)
    # 94 => 3:2, 4:20, 5:72
    save_as([bk_syms,sym_cases,sym_ids],'bk_syms.gol')
    # reduce env symsets 

bk_syms,cases,ids = load_data('bk_syms.gol')