
from auxs import *
from block import *
import os
import networkx as nx
<<<<<<< HEAD
=======
from networkx.drawing.nx_agraph import to_agraph
import seaborn as sns
import pandas as pd
from pyemd import emd
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
import pdb

# block
block = mk_gol_pattern('block')

# skip running single exps
run_exps = False

# 1) proto block domains no constraints
if run_exps==True:
    xpn = True
    e0 = False
    ct = False
    print('\n1) e0 = {}'.format(e0))
    if 'pb_domains.gol' in os.listdir():
        pb_domains = load_data(filename='pb_domains.gol')
        print_ac_cases(pb_domains,title='proto-block domains:')
    else:
        pb_domains = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
        save_as(pb_domains,name='pb_domains',ext='gol')
# 1331/65536

# 2) proto block domains e0=True:
if run_exps==True:
    xpn = True
    e0 = True
    ct = False
    print('\n2) e0 = {}'.format(e0))
    if 'pb_domains_e0.gol' in os.listdir():
        pb_domains_e0 = load_data(filename='pb_domains_e0.gol')
        print_ac_cases(pb_domains_e0,title='proto-block (e0) domains:')
    else:
        pb_domains_e0 = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
        save_as(pb_domains_e0,name='pb_domains_e0',ext='gol')
# 523
# same as e0=False until ac=5, small change for 6, big decay later

# 3) proto block domains expanded=False
# e0 = False
# xpn = False
# ct = False
# print('\n3) expanded = {}'.format(xpn))
# pb_domains_3 = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
# 1459
# grows up???
# because 3 is too weird and theoretically worse
# i'll keep expanded=True
xpn = True

# 4.1) apply ct to 1)
if run_exps==True:
    e0 = False
    ct = True
    print('\n4.1) e0 = {}, ct = {}'.format(e0,ct))
    if 'pb_domains_ct.gol' in os.listdir():
        pb_domains_ct = load_data(filename='pb_domains_ct.gol')
        print_ac_cases(pb_domains_ct,title='proto-block (ct) domains:')
    else:
        pb_domains_ct = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
        save_as(pb_domains_ct,name='pb_domains_ct',ext='gol')
# 1331 -> 1195
# affects only cases < 7

# 4.2) apply ct to 2)
if run_exps==True:
    e0 = True
    ct = True
    print('\n4.2) e0 = {}, ct = {}'.format(e0,ct))
    if 'pb_domains_e0_ct.gol' in os.listdir():
        pb_domains_e0_ct = load_data(filename='pb_domains_e0_ct.gol')
        print_ac_cases(pb_domains_e0_ct,title='proto-block (e0_ct) domains:')
    else:
        pb_domains_e0_ct = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
        save_as(pb_domains_e0_ct,name='pb_domains_e0_ct',ext='gol')
# 523 -> 387
# same as 4.1) for ac=3,4,5. Then, fast decay

# 5.1) apply symmetries to 4.1)
if run_exps==True:
    e0 = False
    ct = True
    sym = True
    print('\n5.1) e0 = {}, ct = {}, sym = {}'.format(e0,ct,sym))
    # pb_domains = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
    if 'pb_domains_ct_sms.gol' in os.listdir():
        pb_sms = load_data(filename='pb_domains_ct_sms.gol')
        print_ac_cases(pb_sms,title='pb_domains_ct_sms:')
    else:
        pb_sms = mk_symsets(pb_domains_ct,block,incremental=True,print_data=True)
        save_as(pb_sms,name='pb_domains_ct_sms',ext='gol')
# 1331 -> 1195 -> 110

# 5.2) apply symmetries to 4.2)
if run_exps==True:
    e0 = True
    ct = True
    print('\n5.2) e0 = {}, ct = {}, symsets'.format(e0,ct))
    # pb_domains = get_sxs_from_sy(block,'block',e0=e0,ct=ct,xpn=xpn,print_data=True)
    if 'pb_domains_e0_ct_sms.gol' in os.listdir():
        pb_sms_e0 = load_data(filename='pb_domains_e0_ct_sms.gol')
        print_ac_cases(pb_sms_e0,title='pb_domains_e0_ct_sms:')
    else:
        pb_sms_e0 = mk_symsets(pb_domains_e0_ct,block,incremental=True,print_data=True)
        save_as(pb_sms_e0,name='pb_domains_e0_ct_sms',ext='gol')
# 523 -> 387 -> 13

# 6.1) apply is_in_domain to 5.1)
if run_exps==True:
    print('\n6.1) minsets: e0 = {}, ct = {}, sym = {}'.format(e0,ct,sym))
    pb_minset = mk_minset(pb_sms,print_data=True)
# 1331 -> 1195 -> 110 -> 13

# 6.2) apply is_in_domain to 5.2)
if run_exps==True:
    print('\n6.2) minsets: e0 = {}, ct = {}, sym = {}'.format(e0,ct,sym))
    pb_minset_e0 = mk_minset(pb_sms_e0,print_data=True) 
# 523 -> 387 -> 13 -> 12 

# pb_minset and pb_minset_e0 are almost the same, except that:
# i) pb_minset excludes block as a proto block (pb0 is more basic)
# ii) it adds an extra, similar to another, but yielding e1
# so if we avoid e0 and consider blocks in different doms
# i) the block itself wouldn't be a primary proto-block
# ii) there would be 13 proto-blocks

# 7) do the same, one step back for pb0
if run_exps==True:
    # params
    xpn = True
    e0 = False
    ct = True
    # proto block 0
    print('\n7) proto block 0:')
    pb0 = mk_gol_pattern('pb0')
    # domains
    if 'pb0_domains.gol' in os.listdir():
        pb0_domains = load_data(filename='pb0_domains.gol')
        print_ac_cases(pb0_domains,title='proto-block-0 domains:')
    else: 
        pb0_domains = get_sxs_from_sy(pb0,'pb0',e0=e0,ct=ct,xpn=xpn,print_data=True)
        save_as(pb0_domains,name='pb0_domains',ext='gol')
    # 2864/65536 -> ct: 2712
    # symmetries
    if 'pb0_sms.gol' in os.listdir():
        pb0_sms = load_data(filename='pb0_sms.gol')
        print_ac_cases(pb0_sms,title='proto-block-0 symmetries:')
    else: 
        pb0_sms = mk_symsets(pb0_domains,pb0,incremental=True,print_data=True)
        save_as(pb0_sms,name='pb0_sms',ext='gol')
    # 2864/65536 -> ct: 2712 -> sms: 316
    # minset
    if 'pb0_minset.gol' in os.listdir():
        pb0_ms = load_data(filename='pb0_minset.gol')
        print_ac_cases(pb0_ms,title='proto-block-0 minset:')
    else: 
        pb0_ms = mk_minset(pb0_sms,print_data=True)
        save_as(pb0_ms,name='pb0_minset',ext='gol')
# 2864/65536 -> ct: 2712 -> sms: 316 -> minset: 19

# 8) Same but for the rest of the basic proto blocks 

# General function
def mk_proto_px(px_name,xpn=True,e0=False,ct=True,print_data=True):
    px = mk_gol_pattern(px_name)
    # domains
    file_dxs = 'proto-{}_domains.gol'.format(px_name)
    if file_dxs in os.listdir():
        px_dxs = load_data(filename=file_dxs)
        print_ac_cases(px_dxs,title='proto-{} domains:'.format(px_name))
    else: 
        px_dxs = get_sxs_from_sy(px,px_name,xpn=xpn,e0=e0,ct=ct,print_data=print_data)
        save_as(px_dxs,name='proto-{}_domains'.format(px_name),ext='gol')
    # symsets
    file_sms = 'proto-{}_symsets.gol'.format(px_name)
    if file_sms in os.listdir():
        px_sms = load_data(filename=file_sms)
        print_ac_cases(px_sms,title='proto-{} symsets:'.format(px_name))
    else:
        px_sms = mk_symsets(px_dxs,px,incremental=True,print_data=print_data)
        save_as(px_sms,name='proto-{}_symsets'.format(px_name),ext='gol')
    # minsets
    file_min = 'proto-{}_minset.gol'.format(px_name)
    if file_min in os.listdir():
        px_min = load_data(filename=file_min)
        print_ac_cases(px_min,title='proto-{} minset:'.format(px_name))
    else:
        px_min = mk_minset(px_sms,print_data=print_data)
        save_as(px_min,name='proto-{}_minset'.format(px_name),ext='gol')
    # graph and data
    pxg = nx.DiGraph()
    return px,px_min

# 9) As objects:

# Summary (expanded=True, e0=False):
# protoblocks
# print('\n\nblock\n')
block = mk_gol_pattern('block')
# block = GolPx('block')
# 1331/65536 -> ct: 1195 -> gensms: 145 -> sms: 110 -> min: 13
# proto-pb0 (pacman)
# print('\n\npb0\n')
# pb0 = mk_gol_pattern('pb0')
# pb0 = GolPx('pb0')
# 2864/65536 -> ct: 2712 -> gensms: 358 -> sms: 316 -> min: 19
# proto-pb1 (snake)
# proto-pb2 (helix)

# 10) check sx -> sy, sx = active cells + membrane
if False==True:
    pb0 = expand_domain(pb0)
    pb0_dxs = mk_sx_domains('pb0',membrane=True)
    pb0_dxs,sxys = get_sxys_from_sx(pb0_dxs,pb0)
    # mk basic symsets
    ids35 = sum_in_range(sxys,3,5)
    sms35,sms35_cases,sms35_ids = mk_symsets(sxys[ids35],pb0,incremental=True,return_data=True)
    # save symsets only (23)
    sxys[ids35] = 0
    sxys[ids35[sms35_ids]] = sms35.reshape(sms35.shape[0],sms35.shape[1]*sms35.shape[2])
    # do 6 (212)
    ids6 = sum_is(sxys,6)
    sms6,sms6_cases,sms6_ids = mk_symsets(sxys[ids6],pb0,incremental=False,return_data=True)
    sxys[ids6] = 0
    sxys[ids6[sms6_ids]] = sms6.reshape(sms6.shape[0],sms6.shape[1]*sms6.shape[2])
    # apply minsets

# minimal proto blocks
# e0 = False, ct = True, symsets, minsets + adjacencies
if run_exps:
    minset_pbs = load_data(filename='proto-block_minset.gol')
    pbs = check_adjacency(minset_pbs,arrays=True)
    save_as(pbs,'proto-block_pbs',ext='gol')

# forward proto blocks
if run_exps:
    pbs = load_data(filename='proto-block_pbs.gol')
    pb0 = pbs[0]
    pb0_dxs = mk_sx_domains(pbs[0])
    pb0_dxs,pb0_sxys = get_sxys_from_sx(pb0_dxs,pb0,decay_txs=1)
    # nonzero: 4088/4096, after CT: 3519, after decay=1: 2974 
    pb0_sxys = rm_non_env(pb0_sxys,pb0)
    # this filter doesn't change the total amount, but changes the ca distribution
    pb0_sms = mk_symsets(pb0_sxys,pb0,incremental=True)
    # general symsets: 499, incremental symsets: 413
    pb0_adj = check_adjacency(pb0_sms)
    # after adjacency (discard discontinuous compositions): 264
    pb0_ftb = check_basic_patterns(pb0_adj)
    # after filtering out basic patterns: 205
    
    # pb0_mms = mk_minset(pb0_sms)
    # basic/minimal symsets (not contained by others): 33
    # save_as(pb0_mms,'fwd_pb0.gol')

# pb0_adj = load_data(filename='pb0_fwd_adj.gol')
# pb0_ftb = check_basic_patterns(pb0_adj)
# pb0_sms = load_data(filename='pb0_fwd_sms.gol')
# pb0_adj = check_adjacency(pb0_sms)
# pb0_mms = mk_minset(pb0_adj)
# pb0_mms = mk_minset(pb0_sms)
# pb0_fwd = check_adjacency(pb0_mms)

#################################################################

#
# 1: final sequence of methods, for protoblocks
# not the optimal (fastest), but the most ordered way
#
def proto_blocks_fx():
    # 0) define block and all proto-domains
    block = mk_gol_pattern('block')
    block_proto_domains = mk_proto_domains(block)
    # 1) get all proto-domains
    if 'block_pb_domains_ids.gol' in os.listdir():
        pb_domains,pb_ids = load_data('block_pb_domains_ids.gol')
        print_ac_cases(pb_domains,title='valid proto domains (sxs->sy):')
    else:
        pb_domains,pb_ids = mk_sxs_from_sy(block,block_proto_domains)
        save_as([pb_domains,pb_ids],'block_pb_domains_ids.gol')
    # 2) filtering:
    if 'block_ft_pb_domains_ids.gol' in os.listdir():
        ft_pb_domains,ft_ids = load_data('block_ft_pb_domains_ids.gol')
        print_ac_cases(ft_pb_domains,title='filtered proto domains:')
    else:
        # 2.1) Transitional continuity (CT)
        pb_domains_ct,ct_ids = apply_ct(pb_domains,block,ct_ids=True,print_data=True)
        ct_ids = mk_matching_ids(pb_ids,ct_ids)
        # 2.2) Structural continuity (CT)
        pb_domains_st,st_ids = check_adjacency(pb_domains_ct,block,ids=True)
        st_ids = mk_matching_ids(ct_ids,st_ids)
        # 2.3) Continuous, but compositional cases (<2)
        ft_pb_domains,ft_ids = check_basic_patterns(pb_domains_st,block,dx_div=2,ids=True)
        ft_ids = mk_matching_ids(st_ids,ft_ids)
        # save_as([ft_pb_domains,ft_ids],'block_ft_pb_domains_ids.gol')
    # 3) Categorization
    fname = 'block_pbs_txs.gol'
    if fname in os.listdir():
        pbs,pb_txs = load_data(fname)
        print_ac_cases(pbs,title='proto-blocks:')
    else:
        # 3.1) symsets
        pb_sms,sms_cases = mk_dxs_symsets(ft_pb_domains,block,ids=True)
        sms_cases = mk_matching_ids(ft_ids,sms_cases)
        sms_ids = np.array(sorted(list(set(sms_cases[:,0]))))
        # 3.2) minimal sets
        min_pbs,min_cases = mk_minimal_sets(pb_sms,ids=True)
        min_cases = mk_matching_ids(sms_ids,min_cases)
        # 3.3) pb transitions
        pb_txs = get_minsyms_counts(sms_ids,sms_cases,min_cases)
        save_as([min_pbs,pb_txs],fname)
<<<<<<< HEAD
        # 4) Cause repertoire
        pb_crep = pb_txs[:,1]/np.sum(pb_txs[:,1])
        pb_names = ['pb{}'.format(i) for i in range(pb_txs.shape[0])]
        # fig,ax = plt.subplots()
        sns.set(style='darkgrid')
        plt.bar(pb_names,pb_crep,alpha=0.5)
        plt.plot(pb_crep)
        plt.show()
=======
    # 4) Cause repertoire
    pb_crep = pb_txs[:,1]/np.sum(pb_txs[:,1])
    pb_names = ['pb{}'.format(i) for i in range(pb_txs.shape[0])]
    sns.set(style='darkgrid')
    plt.bar(pb_names,pb_crep,alpha=0.5)
    plt.plot(pb_crep)
    plt.show()
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    return min_pbs,pb_txs
# 
# 2: final sequence of methods for fwd-protoblocks
# 
<<<<<<< HEAD
if True:
    proto_blocks,pb_txs = load_data('block_pbs_txs.gol')
    for pi,pb in enumerate(proto_blocks):
        # 0) define proto-blocks and env pb domains
        pb_dxs = mk_sx_domains(pb)
        if pb_dxs.shape[1] > pb.flatten().shape[0]:
            pb = expand_domain(pb)
        print('\nsx domains: {}'.format(pb_dxs.shape[0]))
        # 1) get all nonzero fwd domains (pb is always expanded)
        pb_dxs,pb_sxys,sxys_ids = mk_sxys_from_sx(pb,pb_dxs)
        exp_pb = expand_domain(pb)
        # 2) Filtering
        pb_sxys_ct,ct_ids = apply_ct(pb_sxys,exp_pb,ct_ids=True,print_data=True)
        # 2.1) remove decaying y->z patterns
        # pb_sxys_yz,yz_ids = mk_yz_decay(pb_sxys,exp_pb,ids=True)
        pb_sxys_yz,yz_ids = mk_yz_decay(pb_sxys_ct,exp_pb,ids=True)
        # 2.2) transitional CT
        # pb_sxys_ct,ct_ids = apply_ct(pb_sxys_yz,exp_pb,ct_ids=True,print_data=True)
        # pb_dxs = pb_dxs[ct_ids]
        # 2.3) structural CT
        pb_sxys_st,st_ids = check_adjacency(pb_sxys_ct,exp_pb,ids=True)
        # pb_dxs = pb_dxs[st_ids]
        # 2.4) remove decaying non-env cells (ids don't change)
        pb_sxys_ne = rm_non_env(pb_sxys_st,exp_pb)
        # 2.5) remove compositional patterns
        pb_sxys_ft,ft_ids = check_basic_patterns(pb_sxys_ne,exp_pb,dx_div=2,ids=True)
        # pb_dxs = pb_dxs[ft_ids]
        if pi==1:
            save_as([pb,pb_sxys_ft,ft_ids],'pb1_sxys_ids.gol')
            break
        
            
        # 3) Categorization
        # 3.1) symsets
        # fname = 'pb{}_fwd_symsets_cases.gol'.format(pi)
        # if fname in os.listdir():
        #     pb_fwd_sms,sms_cases = load_data(fname)
        #     print_ac_cases(pb_fwd_sms,title='pb{} forward symsets:'.format(pi))
        # else:
        #     pb_fwd_sms,sms_cases = mk_dxs_symsets(pb_sxys_ft,pb,ids=True)
        #     sms_cases = mk_matching_ids(ft_ids,sms_cases)
        #     save_as([pb_fwd_sms,sms_cases],fname)
        # sms_ids = np.array(sorted(list(set(sms_cases[:,0]))))
        # # 3.2) minimal sets
        # fname = 'pb{}_fwd_txs.gol'.format(pi)
        # if fname in os.listdir():
        #     pb_fwd,pb_fwd_txs = load_data(fname)
        #     print_ac_cases(pb_fwd,title='pb{} forward mins:'.format(pi))
        # else:
        #     pb_fwd,min_cases = mk_minimal_sets(pb_fwd_sms,ids=True)
        #     min_cases = mk_matching_ids(sms_ids,min_cases)
        #     # 3,3) fwd transitions
        #     pb_fwd_txs = get_minsyms_counts(sms_ids,sms_cases,min_cases)
        #     save_as([pb_fwd,pb_fwd_txs],fname)
        # print(pb_fwd)

    

# remove decaying non-environmental cells (changes in CA counts, but no change in ids)
    # pb_domains_ne = rm_non_env(pb_domains,block)

=======
# def fwd_protoblocks_fx(n=0):
proto_blocks,pb_txs = load_data('block_pbs_txs.gol')
fname_fx = 'pbs_fwd_pxs_txs.gol'
if fname_fx in os.listdir():
    fwd_pxs,fwd_txs = load_data(fname_fx)
else:
    fwd_pxs,fwd_txs = [],[]
    for pi,pb in enumerate(proto_blocks):
        print()
        print(pi)
        print(pb)
        print()
        fname = 'pb{}_fwd_ft_dxs.gol'.format(pi)
        if fname in os.listdir():
            pb,pb_sxys_ft,ft_ids = load_data(fname)
        else:
            # 0) define proto-blocks and env pb domains
            # pb = proto_blocks[pi]
            pb_dxs = mk_sx_domains(pb)
            if pb_dxs.shape[1] > pb.flatten().shape[0]:
                pb = expand_domain(pb)
            print('\nsx domains: {}'.format(pb_dxs.shape[0]))
            # 1) get all nonzero fwd domains (pb is always expanded)
            pb_dxs,pb_sxys,sxys_ids = mk_sxys_from_sx(pb,pb_dxs)
            # 2) Filtering
            # 2.1) remove decaying non-env cells (ids don't change)
            pb_sxys_ne,ne_ids = rm_non_env(pb_sxys,pb,ids=True)
            # 2.2) remove decaying y->z patterns
            pb_sxys_yz,yz_ids = mk_yz_decay(pb_sxys_ne,pb,ids=True)
            # 2.3) transitional CT
            pb_sxys_ct,ct_ids = apply_ct(pb_sxys_yz,pb,ct_ids=True,print_data=True)
            # 2.3) structural CT
            pb_sxys_st,st_ids = check_adjacency(pb_sxys_ct,pb,ids=True)
            # 2.5) remove compositional patterns cpx<3
            pb_sxys_ft,cp_ids = rm_composed_dv2(pb_sxys_st,pb,ids=True)
            ft_ids = match_ids_sequence([sxys_ids,ne_ids,yz_ids,ct_ids,st_ids,cp_ids])
            save_as([pb,pb_sxys_ft,ft_ids],fname)
        # 3) Categorization
        fname = 'pb{}_fwd_min_txs.gol'.format(pi)
        if fname in os.listdir():
            min_fwds,min_fwd_txs = load_data(fname)
        else:
            # 3.1) symsets
            pb_fwd_sxys = center_tensor_sxs(sort_by_sum(mk_dxs_tensor(pb_sxys_ft,pb)))
            sms_dxs,sms_cases = mk_symsets_large_dxs(pb_fwd_sxys)
            # 3.2) minimal sets
            min_fwds,min_cases,minmap = mk_minsets_large_dxs(sms_dxs)
            # 3,3) fwd transitions
            min_fwd_txs = mk_minmap_counts(sms_cases,min_cases,minmap)
            print(min_fwd_txs)
            save_as([min_fwds,min_fwd_txs],fname)
        fwd_pxs.append(min_fwds)
        fwd_txs.append(min_fwd_txs)
        if fname_fx not in os.listdir():
            save_as([fwd_pxs,fwd_txs],fname_fx)
# 4) transition matrix
pb_ids = [mk_binary_index(pb,arrays=False) for pb in proto_blocks]
# filter out discontinuous
for pi,pb_fwd_pxs in enumerate(fwd_pxs):
    fwds,fwd_ids = filter_composed_dxs(pb_fwd_pxs,ids=True)
    fwd_pxs[pi] = fwds
    fwd_txs[pi] = fwd_txs[pi][fwd_ids]
# reshape all equal
for pbi,pb_pxs in enumerate(fwd_pxs):
    if np.sum(pb_pxs)==np.sum(pb_pxs[:,1:-1,1:-1]):
        fwd_pxs[pbi] = pb_pxs[:,1:-1,1:-1]
# mk matching sxs and matching ids
for pxi,pxs in enumerate(fwd_pxs):
    for pi,px in enumerate(pxs):
        for pxi2,pxs2 in enumerate(fwd_pxs[pxi+1:]):
            for pi2,px2 in enumerate(pxs2):
                if are_the_same_sx(px,px2):
                    fwd_pxs[pxi+1+pxi2][pi2] = px
for pb in proto_blocks:
    for pxi,pxs in enumerate(fwd_pxs):
        for pi,px in enumerate(pxs):
            if are_the_same_sx(pb,px):
                fwd_pxs[pxi][pi] = pb
pb_fwd_ids = []
for pxs in fwd_pxs:
    pb_fwd_ids.append([mk_binary_index(px,arrays=False) for px in pxs])
for txi,txs in enumerate(fwd_txs):
    fwd_txs[txi][:,0] = pb_fwd_ids[txi]
fwd_ids = sorted(list(set.union(*map(set,pb_fwd_ids))))
# mk transition matrix
df_txs = np.zeros((len(pb_ids)+1,len(fwd_ids)+1)).astype(int)
df_txs[1:,0] = pb_ids
df_txs[0,1:] = fwd_ids
for pbi,pb_txs in enumerate(fwd_txs):
    for pi,pb_id in enumerate(fwd_txs[pbi][:,0]):
        df_txs[pbi+1,np.where(df_txs[0]==pb_id)] = fwd_txs[pbi][pi][1]
# table, graph and plots
df = pd.DataFrame(data=df_txs[1:,1:], index=df_txs[1:,0], columns=df_txs[0,1:])
pb_mapping = {}
for pi,pbi in enumerate(pb_ids):
    pb_mapping[pbi] = [(pbi,pxi[0]) for pxi in fwd_txs[pi]]
pb_edges = []
for key in pb_mapping.keys():
    pb_edges.extend(pb_mapping[key])
if False:
    gx = nx.MultiDiGraph(pb_edges)
    gx.graph['edge'] = {'arrowsize':'0.6', 'splines':'curved'}
    gx.graph['graph'] = {'scale':'3'}
    self_cxs = [i[0] for i in pb_edges if i[0]==i[1]]
    for cxi in self_cxs:
        gx[cxi][cxi][0]['color'] = 'red'
    gxa = to_agraph(gx)
    gxa.layout('dot')
    gxa.draw('protoblocks_mapping.png')
    ereps = df_txs[1:,1:]
    ereps = ereps/np.sum(ereps,axis=1).reshape(6,1)
    for ei,ex in enumerate(ereps):
        plt.plot(ex,label='pb{}'.format(ei))
    sns.set(style='darkgrid')
    plt.xlabel('possible forward states')
    plt.ylabel('p(y|x=sx)')
    plt.legend(loc="upper right")
    plt.show()
# distance matrix and information
pbs,pb_txs = load_data('block_pbs_txs.gol')
crep = pb_txs[:,1]/np.sum(pb_txs[:,1])
ereps = df_txs[1:,1:]
ereps = ereps/np.sum(ereps,axis=1).reshape(6,1)
# txs = df_txs[1:,1:]
pbu = [pbi for pbi in pb_ids if pbi not in fwd_ids]
df_txs = np.pad(df_txs,((0,0),(0,len(pbu))))
df_txs[0,-len(pbu):] = pbu
dm_ids = df_txs[0]
dm_counts = np.zeros((dm_ids.shape[0],dm_ids.shape[0]))
cx_rep = np.zeros(dm_ids.shape[0])
ex_reps = np.pad(ereps,((0,0),(0,len(pbu))))
for fwi,fwd_id in enumerate(fwd_ids):
    if fwd_id in pb_ids:
        dm_counts[fwi] = df_txs[np.where(df_txs[:,0]==fwd_id)]
        cx_rep[fwi] = crep[np.where(np.array(pb_ids)==fwd_id)]
dm_counts = dm_counts[1:,1:]
cx_rep = cx_rep[1:]
dmx,dmy = make_dms(dm_counts)
dmx = dmx/np.sum(dmx)
# pb_names = ['pb{}'.format(i) for i in range(pb_txs.shape[0])]
# sns.set(style='darkgrid')
# plt.bar(pb_names,crep,alpha=0.5)
# plt.plot(crep)
# plt.show()
# information
pdb.set_trace()
for ex_rep in ex_reps:
    enac_info = emd(cx_rep,ex_rep,dmx)
    print(enac_info)
# cx_info = emd(cx_rep,np.ones(cx_rep.shape[0])/cx_rep.shape[0],dmy)
# print(cx_info)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12


