from gol_fxs import *

# 1) how many blinkers survive
# blinker_ys = sx_to_sx(blinker)
# blinker: blinker active cells + membrane

# blinker
blinker,blinker_dx = mk_gol_pattern('blinker',domain=True)
# blinker_dx = expand_domain(blinker)
# pb0
pb0,pb0_dx = mk_gol_pattern('pb0',domain=True)
# pb0_dx = expand_domain(pb0)
# block
block,block_dx = mk_gol_pattern('block',domain=True)
# block_dx = expand_domain(block)
# tetris L
tetrisL,tetrisL_dx = mk_gol_pattern('tetrisL',domain=True)
# tetrisL_dx = expand_domain(tetrisL)
# tetris T
tetrisT,tetrisT_dx = mk_gol_pattern('tetrisT',domain=True)
# yields only projections > 6
# glider 
gliderA,gliderA_dx = mk_gol_pattern('gliderA',domain=True)
# gliderA_dx = expand_domain(gliderA)
# zigzag
zigzag,zigzag_dx = mk_gol_pattern('zigzag',domain=True)
# zigzag_dx = expand_domain(zigzag)
# bar
bar,bar_dx = mk_gol_pattern('bar',domain=True)
# bar_dx = expand_domain(bar)
# tetris Z
tetrisZ,tetrisZ_dx = mk_gol_pattern('tetrisZ',domain=True)
# yields inly projections > 6
# baby like shape
baby,baby_dx = mk_gol_pattern('baby',domain=True)
# baby_dx = expand_domain(baby)
# glider 2
gliderB,gliderB_dx = mk_gol_pattern('gliderB',domain=True)
# gliderB_dx = expand_domain(gliderB)
# flag
flag,flag_dx = mk_gol_pattern('flag',domain=True)
# flag_dx = expand_domain(flag)
# kyte
kyte,kyte_dx = mk_gol_pattern('kyte',domain=True)
# worm
worm,worm_dx = mk_gol_pattern('worm',domain=True)

# min sx projection to known variants
# how many sx subsist after a gol-step 
def sx_to_sx(sx,sx_variants=[]):
    sx_dx = expand_domain(sx)
    sx_dxs = mk_env_binary_domains(sx,membrane=True)
    sx_dxs,sx_dxys = vl_multi_gol_step(sx_dxs,sx_dx,rm_zeros=True)
    sx_dxys = check_act_ct(sx_dxys,sx_dx,ids=False)
    sx_variants = [sx] if len(sx_variants)==0 else sx_variants
    sx_ys = []
    for sxi in sx_variants:
        sxi_ys = is_sx_in_dxs(sxi,sx_dxys,sx_dx,membrane=True,moore_nb=False)
        sx_ys.append(sxi_ys)
    return sx_ys

# fwd transition and fwd symsets
def mk_fwd(sx_name,cap=5,symset_cap=5,save_data=False):
    # def
    sx = mk_gol_pattern(sx_name,variants=False)
    sx_dx = expand_domain(sx)
    sx_dxs = mk_env_binary_domains(sx,membrane=True)
    # mk tx
    sx_dxs,sx_dxys = vl_multi_gol_step(sx_dxs,sx_dx,rm_zeros=True)
    sx_dxys,ct_ids = apply_ct(sx_dxys,sx_dx,rm_zeros=True,ct_ids=True)
    sx_dxs = sx_dxs[ct_ids]
    # cap
    cap_ids = sum_lower(sx_dxys,cap+1)
    sx_dxs = sx_dxs[cap_ids]
    sx_dxys = sx_dxys[cap_ids]
    print_ac_cases(sx_dxys,title='after cap')
    # filter
    sx_dxys,ft_ids = rm_non_sx_cells(sx_dxys,sx_dx,ids=True)
    sx_dxs = sx_dxs[ft_ids]
    sx_dxys,yz_ids = mk_yz_decay(sx_dxys,sx_dx,ids=True)
    sx_dxs = sx_dxs[yz_ids]
    # for possible larger dxs,dxys
    if cap>symset_cap:
        ss_cap_ids = sum_lower(sx_dxys,symset_cap+1)
        sx_dxs = sx_dxs[ss_cap_ids]
        sx_dxys = sx_dxys[ss_cap_ids]
    # symsets
    syms,sym_cases,sym_ids = mk_symsets_large_dxs(sx_dxys,sx_dx,membrane=True,ids=True)
    if save_data:
        fname = 'xgol_{}_cap={}.xgol'.format(sx_name,cap)
        save_as([sx_dxs,sx_dxys,syms,sym_cases,sym_ids],fname)
    return sx_dxs,sx_dxys,syms,sym_cases,sym_ids

# bk_xs,bk_ys,bk_syms,bk_cases,bk_ids = mk_fwd('blinker',save_data=True)
bk_xs,bk_ys,bk_syms,bk_cases,bk_ids = load_data('xgol_blinker_cap=5.xgol')
bk_uf = mk_uniform_ids(blinker,'blinker',bk_syms,bk_cases)

# pb0_xs,pb0_ys,pb0_syms,pb0_cases,pb0_ids = mk_fwd('pb0',cap=7,save_data=True)
pb0_xs,pb0_ys,pb0_syms,pb0_cases,pb0_ids = load_data('xgol_pb0_cap=7.xgol')
pb0_uf = mk_uniform_ids(pb0,'pt-block',pb0_syms,pb0_cases)

# bo_xs,bo_ys,bo_syms,bo_cases,bo_ids = mk_fwd('block',cap=7,save_data=True)
bo_xs,bo_ys,bo_syms,bo_cases,bo_ids = load_data('wgol_block_cap=7.wgol')
bo_uf = mk_uniform_ids(block,'block',bo_syms,bo_cases)

# tl_xs,tl_ys,tl_syms,tl_cases,tl_ids = mk_fwd('tetrisL',cap=5,save_data=True)
tl_xs,tl_ys,tl_syms,tl_cases,tl_ids = load_data('xgol_tetrisL_cap=5.xgol')
tl_uf = mk_uniform_ids(tetrisL,'tetris-L',tl_syms,tl_cases)

# tt_xs,tt_ys,tt_syms,tt_cases,tt_ids = mk_fwd('tetrisT',cap=7,save_data=True)
tt_xs,tt_ys,tt_syms,tt_cases,tt_ids = load_data('xgol_tetrisT_cap=7.xgol')
tt_uf = mk_uniform_ids(tetrisT,'tetris-T',tt_syms,tt_cases)

# gla_xs,gla_ys,gla_syms,gla_cases,gla_ids = mk_fwd('gliderA',cap=5,save_data=True)
gla_xs,gla_ys,gla_syms,gla_cases,gla_ids = load_data('xgol_gliderA_cap=5.xgol')
gla_uf = mk_uniform_ids(gliderA,'glider-A',gla_syms,gla_cases)

# zz_xs,zz_ys,zz_syms,zz_cases,zz_ids = mk_fwd('zigzag',cap=5,save_data=True)
zz_xs,zz_ys,zz_syms,zz_cases,zz_ids = load_data('xgol_zigzag_cap=5.xgol')
zz_uf = mk_uniform_ids(zigzag,'zigzag',zz_syms,zz_cases)

# bar_xs,bar_ys,bar_syms,bar_cases,bar_ids = mk_fwd('bar',cap=7,save_data=True)
# results of cap=7 are strangely too high (more than 5,000 symsets)
bar_xs,bar_ys,bar_syms,bar_cases,bar_ids = load_data('xgol_bar_cap=5.xgol')
bar_uf = mk_uniform_ids(bar,'bar',bar_syms,bar_cases)

# tz_xs,tz_ys,tz_syms,tz_cases,tz_ids = mk_fwd('tetrisZ',cap=7,save_data=True)
tz_xs,tz_ys,tz_syms,tz_cases,tz_ids = load_data('xgol_tetrisZ_cap=7.xgol')
tz_uf = mk_uniform_ids(tetrisZ,'tetris-Z',tz_syms,tz_cases)

# bb_xs,bb_ys,bb_syms,bb_cases,bb_ids = mk_fwd('baby',cap=7,save_data=True)
# bb_xs,bb_ys,bb_syms,bb_cases,bb_ids = load_data('xgol_baby_cap=7.xgol')
bb_xs,bb_ys,bb_syms,bb_cases,bb_ids = load_data('xgol_baby_cap=5.xgol')
bb_uf = mk_uniform_ids(baby,'baby',bb_syms,bb_cases)

# glb_xs,glb_ys,glb_syms,glb_cases,glb_ids = mk_fwd('gliderB',cap=5,save_data=True)
glb_xs,glb_ys,glb_syms,glb_cases,glb_ids = load_data('xgol_gliderB_cap=5.xgol')
glb_uf = mk_uniform_ids(gliderB,'glider-B',glb_syms,glb_cases)

# flag_xs,flag_ys,flag_syms,flag_cases,flag_ids = mk_fwd('flag',cap=7,save_data=True)
flag_xs,flag_ys,flag_syms,flag_cases,flag_ids = load_data('xgol_flag_cap=7.xgol')
flag_uf = mk_uniform_ids(flag,'flag',flag_syms,flag_cases)

# kite_xs,kite_ys,kite_syms,kite_cases,kite_ids = mk_fwd('kite',cap=5,save_data=True)
kite_xs,kite_ys,kite_syms,kite_cases,kite_ids = load_data('xgol_kite_cap=5.xgol')
kite_uf = mk_uniform_ids(kite,'kite',kite_syms,kite_cases)

# worm_xs,worm_ys,worm_syms,worm_cases,worm_ids = mk_fwd('worm',cap=5,save_data=True)
worm_xs,worm_ys,worm_syms,worm_cases,worm_ids = load_data('xgol_worm_cap=5.xgol')
worm_uf = mk_uniform_ids(worm,'worm',worm_syms,worm_cases)


ufs = [bk_uf,pb0_uf,bo_uf,tl_uf,tt_uf,gla_uf,zz_uf,bar_uf,tz_uf,bb_uf,glb_uf,flag_uf,kite_uf,worm_uf]
uf0 = [uf[0] for uf in ufs]
ox = mk_omap_sxs(uf0)
for ufi in ufs:
    ox = extend_omap(ox,ufi)

# mk_ox_graph(ox)
# oxc = deepcopy(ox)
# mk_ox_graph(oxc,min_ny=3)
# oxc = deepcopy(ox)
# mk_ox_graph(oxc,closed=True)
# mk_ox_graph(ox,closed=True)

# ufsp = [pb0_uf,bo_uf,tl_uf,tt_uf,gla_uf,zz_uf,bar_uf,tz_uf,bb_uf,flag_uf,kite_uf,worm_uf]
# plot_patterns(ufsp)
# import pdb;pdb.set_trace()
# mk_env_distinctions(zigzag_dx,zz_xs,zz_ids,sx_name='zigzag',mk_plots=True)
# mk_env_distinctions(baby_dx,bb_xs,bb_ids,sx_name='baby',mk_plots=True)

# mk_env_plots(blinker_dx,bk_xs,bk_ids)
# mk_env_plots(pb0_dx,pb0_xs,pb0_ids)
# mk_env_plots(block_dx,bo_xs,bo_ids)
# mk_env_plots(tetrisL_dx,tl_xs,tl_ids)
# mk_env_plots(tetrisT_dx,tt_xs,tt_ids)
# mk_env_plots(gliderA_dx,gla_xs,gla_ids)
# mk_env_plots(zigzag_dx,zz_xs,zz_ids)
# mk_env_plots(gliderB_dx,glb_xs,glb_ids)
# mk_env_plots(pb0_dx,pb0_xs,pb0_ids)
