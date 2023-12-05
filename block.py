from auxs import *
from gol_fxs import *
from block import *
import os
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import seaborn as sns
import pandas as pd
from pyemd import emd
import pdb

# block
block = mk_gol_pattern('block')
pb0 = mk_gol_pattern('pb0')

def mk_proto_fx(sx,sx_name='sx'):
    # proto block domains
    pb_domains = mk_proto_domains(sx)
    pb_domains = mk_sxs_from_sy(sx,pb_domains,ids=False)
    # filters
    print('\npb domains: {}'.format(pb_domains.shape[0]))
    pb_domains = apply_ct(pb_domains,sx)
    pb_domains = check_adjacency(pb_domains,sx) # TODO: fix with is_in_dxs fx
    pb_domains = rm_non_env(pb_domains,sx)
    pb_domains_ft = mk_dxs_tensor(pb_domains,sx)
    # classification
    pb_sms,sms_cases = mk_symsets_large_dxs(pb_domains_ft)
    pbs,min_cases,minmap = mk_minsets_large_dxs(pb_sms)
    # filter out composed
    pb_counts = mk_minmap_counts(sms_cases,min_cases,minmap)
    pbs,ft_ids = filter_composed_dxs(pbs,ids=True)
    pb_counts = pb_counts[ft_ids]
    # outp
    pb_crep = pb_counts[:,1]/np.sum(pb_counts[:,1])
    pb_names = ['pb{}'.format(i) for i in range(pb_counts.shape[0])]
    sns.set(style='darkgrid')
    plt.bar(pb_names,pb_crep,alpha=0.5)
    plt.ylabel('p(x|y={})'.format(sx_name))
    plt.xlabel('possible proto states')
    plt.plot(pb_crep)
    plt.show()
    return pbs,pb_counts

fname = 'protoblocks.gol'
if fname in os.listdir():
    pbs,pb_counts = load_data(fname) 
else:
    pbs,pb_counts = mk_proto_fx(block,sx_name='block')
    save_as([pbs,pb_counts],fname)

fname = 'proto_pb0s.gol'
if fname in os.listdir():
    pbs0,pbs0_counts = load_data(fname) 
else:
    pbs0,pbs0_counts = mk_proto_fx(pb0,sx_name='pb0')
    save_as([pbs,pb_counts],fname)


def mk_fwd_fx(sx,sx_name=''):
    sx_dxs = mk_sx_domains(sx)
    print('\nsx domains: {}'.format(sx_dxs.shape[0]))
    # 1) get all nonzero fwd domains (pb is always expanded)
    sx_dxs,sx_sxys,sxys_ids = mk_sxys_from_sx(sx,sx_dxs)
    # 2) Filtering
    # 2.1) remove decaying non-env cells (ids don't change)
    sx_sxys,ne_ids = rm_non_env(sx_sxys,sx,ids=True)
    # 2.2) remove decaying y->z patterns
    sx_sxys,yz_ids = mk_yz_decay(sx_sxys,sx,ids=True)
    # 2.3) transitional CT
    sx_sxys,ct_ids = apply_ct(sx_sxys,sx,ct_ids=True)
    # 2.3) structural CT
    sx_sxys,st_ids = check_adjacency(sx_sxys,sx,ids=True)
    # 2.5) remove compositional patterns cpx<3
    sx_sxys,cp_ids = rm_composed_dv2(sx_sxys,sx,ids=True)
    # 3.1) symsets
    sx_fwd_sxys = center_tensor_sxs(sort_by_sum(mk_dxs_tensor(sx_sxys,sx)))
    sms_dxs,sms_cases = mk_symsets_large_dxs(sx_fwd_sxys)
    # 3.2) minimal sets
    fwd_pxs,min_cases,minmap = mk_minsets_large_dxs(sms_dxs)
    # 3,3) fwd transitions
    fwd_counts = mk_minmap_counts(sms_cases,min_cases,minmap)
    print(fwd_pxs)
    print(fwd_counts)
    # outp
    sx_erep = fwd_counts[:,1]/np.sum(fwd_counts[:,1])
    fwd_names = ['pb{}'.format(i) for i in range(fwd_counts.shape[0])]
    sns.set(style='darkgrid')
    plt.bar(fwd_names,sx_erep,alpha=0.5)
    plt.ylabel('p(y|x={})'.format(sx_name))
    plt.xlabel('possible forward states')
    plt.plot(sx_erep)
    plt.show()
    return fwd_pxs,fwd_counts
fwd_pxs,fwd_txs = mk_fwd_fx(block,sx_name='block')