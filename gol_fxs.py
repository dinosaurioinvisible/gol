from gol_auxs import *
from collections import defaultdict as ddd

'''
Domain functions:
1) mk proto domains / mk sx domains (fwd)
2) mk sxs from sy (causes)
3) mk sxys from sx (effects)
Filter functions:
1) rm_non_env
2) mk_yz_decay
3) apply_ct (transitional CT)
4) check_adjacency (structural CT)
5) rm_composed_dv2 (structural CT+)
6) rm_env_cells_dv1
7) rm_env_cells_dv2
8) rm_non_sx_cells
Classification fxs:
1) mk_symsets_large_dxs
2) mk_mynsets_large_dxs
Info fxs: (*)
3) mk_matching_ids/ match_ids_sequence
4) mk_minmap_counts/ get_minsyms_counts
5) mk_txs_matrix
'''

# same, but specific for proto-domains of sx
def mk_proto_domains(sx):
    n_cells = sx.flatten().shape[0]
    return mk_binary_domains(n_cells)
# domain environmental (sx + env) tensor
# given a block, blinker or any other structure from the gol (sx)
# make all the env arrays for sx
def mk_sx_domains(sx):
    # centering and check squared domain 
    sx = mk_min_sqrd_domain(sx,moore_nb=True)
    sx_env = mk_moore_nb(sx)
    sx_binary_dxs = mk_binary_domains(np.sum(sx_env).astype(int))
    non_env_ids = np.where(sx_env.flatten()==0)[0]
    non_env_ids -= np.arange(non_env_ids.shape[0])
    sx_binary_dxs = np.insert(sx_binary_dxs,non_env_ids,1,axis=1)
    non_ids = np.where((sx+sx_env).flatten()==0)
    sx_binary_dxs[:,non_ids] = 0
    return sx_binary_dxs

# get all sx: sx -> sy
# sy: any specific gol pattern domain (block,pb0,etc)
def mk_sxs_from_sy(sy,proto_dxs=[],ids=True,print_data=True):
    if len(proto_dxs)==0:
        proto_dxs = mk_proto_domains(sy)
    sy_vxs = mk_sx_variants(sy)
    for di in tqdm(range(proto_dxs.shape[0])):
        dx = proto_dxs[di].reshape(sy.shape)
        if not is_sx_in_next(sy_vxs,dx):
            proto_dxs[di] = 0
    sxs_ids = sum_nonzero(proto_dxs)
    sxs = proto_dxs[sxs_ids]
    if print_data:
        print_ac_cases(sxs,title='valid proto domains (sxs->sy):')
    if ids:
        return sxs,sxs_ids
    return sxs

# get nonzero sx->sy (sxys) from sx
def mk_sxys_from_sx(sx,sx_dxs=[],ids=True,expanded=True,print_data=True):
    if len(sx_dxs)==0:
        sx_dxs = mk_sx_domains(sx)
    # expanded only makes sense if for sx->sy, if dx(sy) < moore_nb(sy)
    if sum_borders(sx)==0:
        expanded = False
    sxys = multi_gol_step(sx_dxs,sx,expanded=expanded)
    nz_ids = sum_higher(sxys,2)
    # nz_ids = sum_nonzero(sxys)
    sxys = sxys[nz_ids]
    sx_dxs = sx_dxs[nz_ids]
    if print_data:
        print_ac_cases(sxys,title='non zero sxys fwd domains:')
    if ids:
        return sx_dxs,sxys,nz_ids
    return sx_dxs,sxys

'''
Filter functions:
1) rm non env
2) mk yz decay
3) transitional CT
4) structural CT
5) structural CT+
6) rm_env_cells_dv12, dv1 and dv2
7) rm_non_sx_cells
'''

# remove decaying activations within domain (not environment of sx)
# dxs: domains in matrix form
# sx: pattern being analyzed, but only for reshaping
def rm_non_env(dxs,sx,ids=False,print_data=True):
    # if expanded dx
    if sx.flatten().shape[0] < dxs.shape[1]:
        sx = expand_domain(sx)
    dxs_ne = dxs*1
    # for di,dx in enumerate(dxs):
    for di in tqdm(range(dxs.shape[0])):
        dx = dxs[di]
        gx = gol_step(dx.reshape(sx.shape))
        dxs_ne[di] = ((mk_moore_nb(gx)+gx)*dx.reshape(sx.shape)).flatten()
    dxs_ids = sum_higher(dxs_ne,2)
    dxs_ne = dxs_ne[dxs_ids]
    if print_data:
        print_ac_cases(dxs_ne,title='after removing decaying non-env cells:')
    if ids:
        return dxs_ne,dxs_ids
    return dxs_ne

# sxys: domains in matrix form, sx: in matrix form
def mk_yz_decay(sxys,sx,expanded=True,ids=False,print_data=True):
    if sx.flatten().shape[0] < sxys.shape[1]:
        sx = expand_domain(sx)
    if sum_borders(sx) == 0:
        expanded = False
    yz = multi_gol_step(sxys,sx,expanded=expanded)
    yz_ids = sum_higher(yz,2)
    sxys = sxys[yz_ids]
    if print_data:
        print_ac_cases(sxys,title='after yz decay:')
    if ids:
        return sxys,yz_ids
    return sxys

# dxs: matrix of gol domain sts arrays 
# psx: primary pattern/structure determining ct with env=0
def apply_ct(dxs,psx,rm_zeros=True,ct_ids=False,non_ct_ids=False,print_data=True):
    if dxs.shape[1] > psx.flatten().shape[0]:
        psx = expand_domain(psx)
    psx = psx.flatten()
    ids = sum_nonzero(dxs*psx)
    if print_data:
        print_ac_cases(dxs[ids],title='after CT:')
    if ct_ids:
        if rm_zeros:
            return dxs[ids],ids
        return dxs,ids
    if non_ct_ids:
        zero_ids = sum_is(dxs*psx,0)
        if rm_zeros:
            return dxs[ids],zero_ids
        return dxs,zero_ids
    if rm_zeros:
        return dxs[ids]
    return dxs

# discard composed patterns
# dxs = tensor of matrix domains (if not, requires sx for reshaping)
# returns tensor if input is tensor, otherwise matrix
def check_adjacency(dxs,sx=[],print_data=True,ids=False):
    dxs_ids = []
    tensor = True
    if len(dxs.shape) == 2:
        dxs = mk_dxs_tensor(dxs,sx)
        tensor = False
    for di in tqdm(range(dxs.shape[0])):
        dx = dxs[di]
        adj_ct = True
        dx = rm_zero_layers(dx)
        ml = max(max(dx.shape)-min(dx.shape)+1,2)
        dlr,drl = get_diagonals(dx,min_len=ml,sums=True,seps=True)
        if np.sum(dx) != np.sum(rm_isol(dx)):
            adj_ct = False
        elif 0 in np.sum(dx,axis=1) or 0 in np.sum(dx,axis=0):
            adj_ct = False
        elif 0 in [dlr[i]+dlr[i+1] for i in range(len(dlr)-1)]:
            adj_ct = False
        elif 0 in [drl[i]+drl[i+1] for i in range(len(drl)-1)]:
            adj_ct = False
        if adj_ct:
            dxs_ids.append(di)
    dxs_ids = np.array(dxs_ids)
    dxs = dxs[dxs_ids]
    if not tensor:
        if dxs.shape[1]*dxs.shape[2] > sx.flatten().shape[0]:
            sx = expand_domain(sx)
        dxs = dxs.reshape(dxs.shape[0],sx.flatten().shape[0])
    if print_data:
        print_ac_cases(dxs,title='after adjacency:')
    if ids:
        return dxs,dxs_ids
    return dxs

def rm_composed_dv2(dxs,sx,ids=False,print_data=True):
    bp1 = np.zeros((sx.shape))
    bp2 = np.zeros((sx.shape))
    bp1[:2,:2] = np.array((1,1,0,0)).reshape(2,2)
    bp2[:2,:2] = np.array([1,0,0,1]).reshape(2,2)
    bp1 = center_sx_in_dx(bp1)
    bp2 = center_sx_in_dx(bp2)
    bps = np.array([np.rot90(bp1,ri) for ri in range(4)]+[np.rot90(bp2,ri) for ri in range(2)])
    bp_ids = np.ones(dxs.shape[0]).astype(int)
    for bpi in bps:
        bpi_ids = is_sx_in_dxs(bpi,dxs)#,moore_nb=False)
        bp_ids[bpi_ids] = 0
    bp_ids = bp_ids.nonzero()[0]
    dxs = dxs[bp_ids]
    if print_data:
        print_ac_cases(dxs,title='after removing dv2s:')
    if ids:
        return dxs,bp_ids
    return dxs

def rm_env_cells_dv1(dxs,px,ids=False,print_data=True):
    # rm_ids = np.ones(dxs.shape[0])
    for di in tqdm(range(dxs.shape[0])):
        dx = dxs[di].reshape(px.shape)
        if np.sum(dx) != np.sum(rm_isol(dx)):
            dxs[di] = rm_isol(dx).flatten()
            # rm_ids[di] = 0
    rm_ids = sum_higher(dxs,2)
    dxs = dxs[rm_ids]
    if print_data:
        print_ac_cases(dxs,title='after rm isolated 1c')
    if ids:
        return dxs,rm_ids
    return dxs

def rm_env_cells_dv2(dxs,px,ids=False,print_data=True):
    # reshape up, for borders
    dxs = np.pad(dxs.reshape(dxs.shape[0],px.shape[0],px.shape[1]),((0,0),(1,1),(1,1)))
    px = expand_domain(px)
    dxs = dxs.reshape(dxs.shape[0],px.flatten().shape[0])
    dxr = mk_dxs_tensor(dxs,px)
    bx = expand_domain(np.array([[1,1]]))
    vxs = [bx,bx.T]
    cx = expand_domain(np.array([[1,0,0,1]]).reshape(2,2))
    cxs = mk_sx_variants(cx)
    for vx in vxs:
        for wi in range(px.shape[0]-vx.shape[0]+1):
            for wj in range(px.shape[1]-vx.shape[1]+1):
                wx_ids = np.zeros(dxs.shape[0]).astype(int)
                wx = np.zeros((px.shape))
                wx[wi:wi+vx.shape[0],wj:wj+vx.shape[1]] = vx
                wx_ids[sum_is(dxs*wx.flatten(),2)] = 1
                wx_ids[sum_nonzero(dxs*mk_moore_nb(wx).flatten())] = 0
                if np.sum(wx_ids) > 0:
                    dxr[wx_ids.nonzero()[0],wi:wi+vx.shape[0],wj:wj+vx.shape[1]] = 0
    # reshape down
    dxr = dxr[:,1:-1,1:-1]
    px = px[1:-1,1:-1]
    dxr = dxr.reshape(dxr.shape[0],px.flatten().shape[0])
    ft_ids = sum_higher(dxr,2)
    if print_data:
        print_ac_cases(dxr[ft_ids],title='after rm env cells dv2:')
    if ids:
        return dxr[ft_ids],ft_ids
    return dxr[ft_ids]

# this version is slower bur safer
# dxs: domains in matrix from
# px: sx domain for reshaping
def rm_non_sx_cells(dxs,px,ids=False):
    dxs = np.pad(dxs.reshape(dxs.shape[0],px.shape[0],px.shape[1]),((0,0),(1,1),(1,1)))
    dxs = mk_dxs_tensor(dxs,expand_domain(px))
    dxs_ft = dxs*1
    # c1 and c2 non sx cases
    c1 = expand_domain(np.ones((1,1)))
    c2 = expand_domain(np.ones((1,2)))
    d2 = expand_domain(np.array([[1,0,0,1]]).reshape(2,2))
    vs = [c1,c2,c2.T,d2,np.rot90(d2,1)]
    # remove
    for vi in tqdm(range(len(vs))):
        vx = vs[vi]
        for wi in range(dxs.shape[1]-vx.shape[0]+1):
            for wj in range(dxs.shape[2]-vx.shape[1]+1):
                ids1,ids0 = np.zeros(dxs.shape[0]),np.zeros(dxs.shape[0])
                wxs = dxs_ft[:,wi:wi+vx.shape[0],wj:wj+vx.shape[1]]
                ids1[sum_is(wxs*vx,np.sum(vx))] = 1
                ids0[sum_is(wxs*mk_moore_nb(vx),0)] = 1
                # pdb.set_trace()
                idsx = (ids1*ids0).nonzero()[0]
                if idsx.shape[0]>0:
                    # pdb.set_trace()
                    dxs_ft[idsx,wi:wi+vx.shape[0],wj:wj+vx.shape[1]] = 0
                    # dxs_ft[idsx,wi:min(dxs.shape[1]-1,wi+vx.shape[0]),min(dxs.shape[2]-1,wj+vx.shape[1])] = 0
                    
    dxs_ft = dxs_ft[:,1:-1,1:-1]
    dxs_ft = dxs_ft.reshape(dxs_ft.shape[0],px.flatten().shape[0])
    nz_ids = sum_higher(dxs_ft,2)
    print_ac_cases(dxs_ft[nz_ids],title='after rm non sx cells:')
    if ids:
        return dxs_ft[nz_ids],nz_ids
    return dxs_ft[nz_ids]

'''
Classification fxs:
1) mk_symsets_large_dxs
2) mk_mynsets_large_dxs
'''

# final version, faster and more precise
def mk_symsets_large_dxs(dxs,sx=[],membrane=False,ids=False,print_data=True):
    if len(dxs.shape)==2:
        dxs = mk_dxs_tensor(dxs,sx)
    dxs = sort_by_sum(dxs)
    dxs = center_tensor_sxs(dxs)
    # canon/type, if instance, number of instances
    symset_cases = np.ones((dxs.shape[0],2)).astype(int)*-1
    reps = np.zeros(dxs.shape[0])
    if ids:
        symset_ids = {}
    for di in tqdm(range(dxs.shape[0])):
        if reps[di] == 0:
            dx_ids = is_sx_in_dxs(dxs[di],dxs,sx,membrane=membrane)
            symset_cases[di] = [di,dx_ids.shape[0]]
            reps[dx_ids] = 1
            if ids:
                symset_ids[di] = dx_ids
    sms_ids = np.where(symset_cases[:,1]>0)[0]
    dxs = dxs[sms_ids]
    symset_cases = symset_cases[sms_ids]
    if print_data:
        print_ac_cases(dxs,title='symsets:')
    dxs = sort_by_sum(dxs)
    dxs = center_tensor_sxs(dxs)
    if ids:
        return dxs,symset_cases,symset_ids
    return dxs,symset_cases

# also final version, using faster algorithm
# dxs: can be symsets or raw dxs
def mk_minsets_large_dxs(dxs,print_data=True):
    dxs = center_tensor_sxs(sort_by_sum(dxs))
    min_cases = np.ones((dxs.shape[0],2)).astype(int)*-1
    reps = np.zeros(dxs.shape[0])
    minmap = {}
    for di in tqdm(range(dxs.shape[0])):
        if reps[di] == 0:
            dx_ids = is_sx_in_dxs(dxs[di],dxs,moore_nb=False)
            min_cases[di] = [di,dx_ids.shape[0]]
            reps[dx_ids] += 1
            minmap[di] = dx_ids
    min_ids = np.where(min_cases[:,1]>0)[0]
    dxs = dxs[min_ids]
    min_cases = min_cases[min_ids]
    if print_data:
        print_ac_cases(dxs,title='minimal sets:')
    dxs = center_tensor_sxs(sort_by_sum(dxs))
    return dxs,min_cases,minmap


'''
mappings
'''
def mk_uniform_ids(sx,sx_name,syms,cases):
    dix = {}
    dix[0] = {}
    vxs = sorted([array2int(vxi) for vxi in mk_sx_variants(sx,mk_non_env=False)])
    dix[0]['sx'] = sx
    dix[0]['id'] = vxs[0]
    dix[0]['label'] = sx_name
    dix[0]['vxs'] = vxs
    for ei,sym in enumerate(syms):
        dix[ei+1] = {}
        sym = expand_domain(rm_zero_layers(sym))
        vxs = sorted([array2int(vxi) for vxi in mk_sx_variants(sym,mk_non_env=False)])
        dix[ei+1]['sx'] = sym
        dix[ei+1]['id'] = vxs[0]
        dix[ei+1]['label'] = vxs[0]
        dix[ei+1]['dx_id'] = cases[ei][0]
        dix[ei+1]['vxs'] = vxs
        dix[ei+1]['txs'] = cases[ei][1]
    return dix

def mk_uniform_data(sx,sx_dx,dxs,sx_name,syms,cases,ids):
    # 0: sx
    dix = {}
    dix[0] = {}
    vxs = sorted([array2int(vxi) for vxi in mk_sx_variants(sx,mk_non_env=False)])
    dix[0]['sx'] = sx
    dix[0]['sx_dx'] = sx_dx
    dix[0]['id'] = vxs[0]
    dix[0]['label'] = sx_name
    dix[0]['vxs'] = vxs
    # 1...n: sx -> sy
    for ei,sym in enumerate(syms):
        dix[ei+1] = {}
        sym = expand_domain(rm_zero_layers(sym))
        vxs = sorted([array2int(vxi) for vxi in mk_sx_variants(sym,mk_non_env=False)])
        dix[ei+1]['sx'] = sym
        dix[ei+1]['id'] = vxs[0]
        dix[ei+1]['label'] = vxs[0]
        dix[ei+1]['dx_id'] = cases[ei][0]
        dix[ei+1]['vxs'] = vxs
        dix[ei+1]['txs'] = cases[ei][1]
    # env domain
    env_info = mk_env_distinctions(sx_dx,dxs,ids)
    for ei,key in enumerate(ids.keys()):
        emdx,ncases,env,edx,idx_counts,idx_dist = env_info[key]
        dix[ei+1]['envs'] = env
        dix[ei+1]['emd'] = emdx
    return dix

def mk_omap_sxs(ufs0):
    # ox = nx.MultiDiGraph()
    ox = nx.DiGraph()
    for uf in ufs0:
        ox.add_node(uf['id'],label=uf['label'],vxs=uf['vxs'])
    return ox
# def mk_omap(dix):
#     ox = nx.DiGraph()
#     for dk,dv in dix.items():
#         if dv['id'] not in ox.nodes:
#             ox.add_node(dv['id'],label=dv['label'],vxs=dv['vxs'])
#     for dk,dv in dix.items():
#         if dk>0:
#             ox.add_edge(dix[0]['id'],dv['id'],weight=dv['txs'])
#     return ox
def extend_omap(ox,dix):
    for dk,dv in dix.items():
        if dv['id'] not in ox.nodes:
            ox.add_node(dv['id'],label=dv['label'],vxs=dv['vxs'])
    for dk,dv in dix.items():
        if dk>0:
            ox.add_edge(dix[0]['id'],dv['id'],weight=dv['txs'])
    return ox        


    




'''
Information fxs
'''

# to make ids from a reduced domain, match ids from a previous larger domain
# dx_ids: large (original) domain ids (array)
# subset_ids: reduced domain ids (array)
def mk_matching_ids(dx_ids,subset_ids):
    if len(subset_ids.shape)==1:
        return dx_ids[subset_ids]
    if len(subset_ids.shape)==2:
        if subset_ids.shape[0]>subset_ids.shape[1]:
            for ci in range(subset_ids.shape[1]):
                subset_ids[:,ci] = dx_ids[subset_ids[:,ci]]
            return subset_ids
        for ri in range(subset_ids.shape[0]):
            subset_ids[ri] = dx_ids[subset_ids[ri]]
        return subset_ids
def match_ids_sequence(ids_seq):
    mt_ids = ids_seq[0]
    for sqi_ids in ids_seq[1:]:
        mt_ids = mt_ids[sqi_ids]
    return mt_ids
# map min to sms cases for counts and dists
def mk_minmap_counts(sms_cases,min_cases,minmap):
    counts = min_cases*1
    for i in range(min_cases.shape[0]):
        mi_id,mi_sum = min_cases[i]
        mi_sum += np.sum(sms_cases[minmap[mi_id],1])
        counts[i] = [mi_id,mi_sum]
    return counts
# to match symsets and minsets ids
def get_minsyms_counts(sms_ids,sms_cases,min_cases):
    # get ids and counts
    min_ids = np.array(sorted(set(list(min_cases[:,0]))))
    min_counts = np.array([[i,np.where(min_cases[:,0]==i)[0].shape[0]] for i in min_ids])
    sms_counts = np.array([[i,np.where(sms_cases[:,0]==i)[0].shape[0]] for i in sms_ids])
    # replace ids in sms counts for min ids and add to txs
    txs_counts = min_counts*1
    for smi,count in sms_counts:
        if smi not in min_ids:
            smi_min_ids = min_cases[np.where(min_cases[:,1]==smi)][:,0]
            for xi in smi_min_ids:
                txs_counts[np.where(txs_counts[:,0]==xi),1] += count
        else:
            txs_counts[np.where(txs_counts[:,0]==smi),1] += count
    return txs_counts

def mk_txs_matrix(pxs,txs,dims=4):
    # clean discontinuous patterns
    pxtx = []
    for pbi,fpxs in enumerate(pxs):
        for fpi,fpx in enumerate(fpxs):
            if is_composed(fpx):
                fpx_id,px = mk_binary_index(fpx,dims=dims)
                pxtx.append([fpx_id,(pbi,fpi),px,txs[pbi][fpi][1]])
    return pxtx