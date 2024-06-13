import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import pdb
from pyemd import emd
from pyemd import emd_samples

'''
mk_gol_pattern
gol_step/ multi_gol_step
mk_binary_domains
mk_env_binary_domains
int2array / array2int
sum_is/ sum_in_range/ sum_higher/ sum_lower/ sum_nonzero
sort_by_sum
sum_borders
mk_moore_nb
expand_domain/ expand_multiple_domains
mk_min_sqrd_domain
mk_min_sx_variants/ mk_sx_variants
check_act_ct
print_ac_cases
rm_isol/ rm_zero_layers
adjust_domains
mk_dxs_tensor
center_sx_in_dx/ center_tensor_sxs
mk_binary_index
are_the_same_sx
is_sx_in_dx/ is_sx_in_dxs/ is_sx_in_next
filter_composed_dxs/ is_composed
# info related
make_dms
# other
save_as/ load_data
# old fxs:
are_sx_instances/ are_symmetrical
is_block/ is_blinker/ is_protoblock0/ is_block_next
check_translation
rm_env
reduce_to_min_sqr
get_diagonals
is_in_domain
match_dxs_sms_ids
check_basic_patterns
# end old fxs
'''

# make canonical gol patterns (sx,e=0) from word inputs
# minimal form: active cells + moore neighborhood rectangled
def mk_gol_pattern(px,domain=False,variants=False):
    if px == 'block':
        dx = np.zeros((4,4))
        dx[1:-1,1:-1] = 1
        # return dx
    elif px == 'pb0':
        dx = np.ones((2,2))
        dx[0,1] = 0
        dx = np.pad(dx,(1,1))
        # return dx
    elif px == 'pb2':
        dx = np.zeros((2,4))
        dx[1,:2] = 1
        dx[0,2:] = 1
        dx = np.pad(dx,(1,1))
        # return dx
    elif px == 'blinker':
        d1 = np.zeros((5,3))
        d1[1:-1,1] = 1
        if not variants:
            dx = d1
        else:
            d2 = np.ascontiguousarray(d1.T)
            dx = [d1,d2]
        # return dx
    elif px == 'glider':
        dx = []
        d1 = np.zeros((5,5))
        d1[1,2] = 1
        d1[2,3] = 1
        d1[3,1:-1] = 1
        if not variants:
            dx = d1
        else:
            d2 = np.zeros((5,5))
            d2[1,1] = 1
            d2[2:-1,2] = 1
            d2[1:3,3] = 1
            for di in [d1,d2]:
                for ri in range(4):
                    dr = np.rot90(di,ri)
                    dt = np.ascontiguousarray(dr.T)
                    dx.extend([dr,dt])
        # return dx
    elif px == 'gliderA':
        dx = np.zeros((5,5))
        dx[1,2] = 1
        dx[2,3] = 1
        dx[3,1:-1] = 1
        # return d1
    elif px == 'gliderB':
        dx = np.zeros((5,5))
        dx[1,1] = 1
        dx[2:-1,2] = 1
        dx[1:3,3] = 1
        # return d2
    elif px == 'tetrisL':
        dx = expand_domain(np.ones((1,3)))
        dx[2,3] = 1
        dx = expand_domain(rm_zero_layers(dx))
        # return dx
    elif px == 'tetrisT':
        dx = expand_domain(np.ones((1,3)))
        dx[2,2] = 1
        dx = expand_domain(rm_zero_layers(dx))
        # return dx
    elif px == 'zigzag':
        dx = expand_domain(np.array([[1,0,1,0],[0,1,0,1]]))
        # return dx
    elif px == 'bar':
        dx = expand_domain(np.ones((1,4)))
        # return dx
    elif px == 'tetrisZ':
        dx = expand_domain(np.array([[0,1,1],[1,1,0]]))
        # return dx
    elif px == 'baby':
        dx = expand_domain(np.array([[1,0,1],[0,1,1]]))
        # return dx
    elif px == 'flag':
        dx = expand_domain(np.ones((2,2)))
        dx[3,2] = 1
        dx = expand_domain(rm_zero_layers(dx))
        # return dx
    elif px == 'kite':
        dx = expand_domain(np.ones((2,2)))
        dx[3,3] = 1
        dx = expand_domain(rm_zero_layers(dx))
        # return dx
    elif px == 'worm':
        dx = expand_domain(np.array([[0,0,1,0],[1,1,0,1]]))
    else:
        print('\npattern not defined\n')
    if domain:
        return dx,expand_domain(dx)
    return dx

# game of life transition
# expanded adds an external layer
def gol_step(world_st,expanded=False):
    world = world_st*1 if not expanded else expand_domain(world_st)
    world_copy = world*1
    for ei,vi in enumerate(world_copy):
        for ej,vij in enumerate(vi):
            nb = np.sum(world_copy[max(0,ei-1):ei+2,max(ej-1,0):ej+2]) - vij
            vx = 1 if (vij==1 and 2<=nb<=3) or (vij==0 and nb==3) else 0
            world[ei,ej] = vx
    return world
# gol transition for multiple arrays
# sx_domains: gol lattice/domain arrays for each sx -> sy
# sx: matrix form to reshape arrays 
# mk_zero: makes sums<3 = 0 (will die next)
def multi_gol_step(sx_domains,sx,mk_zero=True,expanded=False):
    # shape & output array
    sxys = np.zeros(sx_domains.shape) if not expanded else np.zeros((sx_domains.shape[0],expand_domain(sx).flatten().shape[0]))
    # for larger domains
    if sx_domains.shape[0] > 2**12:
        for di in tqdm(range(sx_domains.shape[0])):
            dx = sx_domains[di].reshape(sx.shape)
            if np.sum(dx)>2:
                sxys[di] = gol_step(dx,expanded=expanded).flatten()
        return sxys
    # simulate transitions
    for di,dx in enumerate(sx_domains):
        if np.sum(dx)>2:
            sxys[di] = gol_step(dx.reshape(sx.shape),expanded=expanded).flatten()
    if mk_zero:
        # remove decaying domains (<3)
        sxys[sum_lower(sxys,3)] = 0
    return sxys
# same, for very large sets
def vl_multi_gol_step(sx_domains,sx,rm_zeros=True):
    sxys = np.zeros((sx_domains.shape))
    for di in tqdm(range(sx_domains.shape[0])):
        dx = sx_domains[di].reshape(sx.shape)
        sxys[di] = gol_step(dx).flatten()
    if rm_zeros:
        nz_ids = sum_higher(sxys,2)
        return sx_domains[nz_ids],sxys[nz_ids]
    return sxys

# a tensor for all binary combinations
# n_cells are all the env cells in domain
def mk_binary_domains(n_cells):
    n_cells = n_cells if type(n_cells)==int else int(n_cells)
    doms = np.zeros((2**n_cells,n_cells)).astype(int)
    for i in range(n_cells):
        f = 2**i
        xi = np.concatenate((np.zeros(f),np.ones(f)))
        n = int(2**n_cells/(2**(i+1)))
        doms[:,-1-i] = np.tile(xi,n)
    return doms
# more general fx, for gol patterns
# domain environmental (sx + env) tensor
# given a block, blinker or any other structure from the gol (sx)
# make all the env arrays for sx
# e_cells are all the cells in the environment
def mk_env_binary_domains(sx,membrane=False,tensor=False):
    sx_dx = expand_domain(rm_zero_layers(sx))
    if membrane:
        sx = expand_domain(sx)
        # sx_dx = expand_domain(np.ones(sx_dx.shape))
        sx_dx = sx+mk_moore_nb(sx)
    sx_env = mk_moore_nb(sx_dx)
    binary_dxs = mk_binary_domains(np.sum(sx_env).astype(int))
    non_env_ids = np.where(sx_env.flatten()==0)[0]
    non_env_ids -= np.arange(non_env_ids.shape[0])
    binary_dxs = np.insert(binary_dxs,non_env_ids,1,axis=1)        
    non_ids = np.where((sx+sx_env).flatten()==0)
    binary_dxs[:,non_ids] = 0
    if tensor:
        binary_dxs = binary_dxs.reshape(binary_dxs.shape[0], sx_dx.shape[0], sx_dx.shape[1])
    return binary_dxs

# int > binary array
def int2array(ni,arr_len,mn=1):
    # reversed for cell order
    x = np.array([int(i) for i in np.binary_repr(ni,arr_len) [::-1]])
    if mn > 1:
        return x.reshape(mn,mn)
    return x
# array to int
def array2int(arr):
    xi = np.sum([x<<e for e,x in enumerate(arr.flatten().astype(int))])
    return xi

# fxs that return ids of the arrays of a matrix that:
# sum = x; higher than x; lower than x; nonzero
def sum_is(matrix,x,axis=1,arrays=False):
    if len(matrix.shape) == 3:
        if arrays:
            return matrix[np.where(np.sum(matrix,axis=(1,2))==x)[0]]
        return np.where(np.sum(matrix,axis=(1,2))==x)[0]
    if arrays:
        return matrix[np.where(np.sum(matrix,axis=axis)==x)[0]]
    return np.where(np.sum(matrix,axis=axis)==x)[0]
def sum_in_range(matrix,rl,rh,axis=1,arrays=False):
    if len(matrix.shape)==3:
        ids = np.array([])
        for i in range(rl,rh):
            ids = np.concatenate((ids,sum_is(matrix,i)))
        if arrays:
            return matrix[ids.astype(int)]
        return ids
    ids = np.where(np.logical_and(np.sum(matrix,axis=axis)>=rl,np.sum(matrix,axis=axis)<rh))[0]
    if arrays:
        return matrix[ids]
    return ids
def sum_higher(matrix,x,axis=1,arrays=False):
    if len(matrix.shape)==3:
        if arrays:
            return matrix[np.where(np.sum(matrix,axis=(1,2))>=x)[0]]
        return np.where(np.sum(matrix,axis=(1,2))>=x)[0]
    if arrays:
        return matrix[np.where(np.sum(matrix,axis=axis)>=x)[0]]
    return np.where(np.sum(matrix,axis=axis)>=x)[0]
def sum_lower(matrix,x,axis=1,arrays=False):
    if len(matrix.shape)==3:
        if arrays:
            return matrix[np.where(np.sum(matrix,axis=(1,2))<x)[0]]
        return np.where(np.sum(matrix,axis=(1,2))<x)[0]
    if arrays:
        return matrix[np.where(np.sum(matrix,axis=axis)<x)[0]]
    return np.where(np.sum(matrix,axis=axis)<x)[0]
def sum_nonzero(matrix,axis=1,arrays=False):
    if len(matrix.shape) == 3:
        if arrays:
            return matrix[np.sum(matrix,axis=(1,2)).nonzero()[0]]
        return np.sum(matrix,axis=(1,2)).nonzero()[0]
    if arrays:
        return matrix[np.sum(matrix,axis=axis).nonzero()[0]]
    return np.sum(matrix,axis=axis).nonzero()[0]

# for matrix/tensor domains
def sort_by_sum(dxs):
    return np.array(sorted(list(dxs),key=lambda x:np.sum(x)))

# to check if some sx is surrended by zeros in some domx 
def sum_borders(domx):
    cx = np.sum(domx[1:-1,1:-1])
    return np.sum(domx)-cx

# make moore neighborhood
# sxr: reduced sx (the min rectangle containing all act. cells)
def mk_moore_nb(sxr):
    # sxr = np.pad(sxr,(1,1))
    moore_nb = sxr*1
    for i in range(sxr.shape[0]):
        for j in range(sxr.shape[1]):
            moore_nb[i,j] = np.sum(sxr[max(0,i-1):i+2,max(0,j-1):j+2])
    moore_nb = moore_nb * np.abs(sxr-1)
    return np.where(moore_nb>0,1.,0.)

# expand domain size: rows/cols=(bef 0,aft n), else layers
def expand_domain(sx,layers=1,rows=(0,0),cols=(0,0)):
    if np.sum([rows,cols])>0:
        return np.pad(sx,(rows,cols),mode='constant')
    return np.pad(sx,layers,mode='constant')
# expand one layer for multiple domains
def expand_multiple_domains(dxs,sx=()):
    ms,ns = sx.shape if len(sx)>0 else [np.sqrt(dxs.shape[1]).astype(int)]*2
    e_dxs = np.zeros((dxs.shape[0],(ms+2)*(ns+2)))
    sx_ids = expand_domain(np.ones((ms,ns))).flatten().nonzero()[0]
    e_dxs[:,sx_ids] = dxs
    return e_dxs

# sx: pattern in domain in matrix form
# min space for pattern + moore nb (deactivatable)
def mk_min_sqrd_domain(sx,moore_nb=True):
    if sx.shape[0] == sx.shape[1]:
        if sum_borders(sx) > 0:
            if moore_nb:
                return expand_domain(sx)
            return sx
        sx = rm_zero_layers(sx,squared=True)
        if moore_nb:
            return expand_domain(sx)
        return sx
    # check if difference is even or not
    ds = abs(int((sx.shape[0]-sx.shape[1])/2))
    ds1,ds2 = ds,ds
    if (sx.shape[0]-sx.shape[1])%2 != 0:
        ds2 += 1
    # fill 
    if sx.shape[0]<sx.shape[1]:
        sx = np.pad(sx,((ds1,ds2),(0,0)))
    else:
        sx = np.pad(sx,((0,0),(ds1,ds2)))
    if moore_nb:
        return expand_domain(sx)
    return sx

# returns rotation and transposition variants for dx
def mk_min_sx_variants(dx,moore_nb=False):
    sx = rm_zero_layers(dx)
    if moore_nb:
        sx = expand_domain(sx)
    sxs,vxs = [],[]
    for ri in range(4):
        sxr = np.rot90(sx,ri)
        sxrt = np.ascontiguousarray(sxr.T)
        sxs.extend([sxr,sxrt])
    for sxi in sxs:
        sxi_in_vxs = False
        for vxi in vxs:
            if np.array_equal(sxi,vxi):
                sxi_in_vxs = True
                break
        if sxi_in_vxs==False:
            vxs.append(sxi)
    return vxs
# make all variants (rotation,transposition,non-env/moore nb)
# this should be mk env or mk non moore nb actually
def mk_sx_variants(sx,mk_non_env=True):
    vxs,sxs = [],[]
    # rotations
    for ri in range(4):
        sxr = np.rot90(sx,ri)
        # transpositions
        sxrt = np.ascontiguousarray(sxr.T)
        vxs.extend([sxr,sxrt])
        # non env/moore different variants
        if mk_non_env:
            sxr_ne_base = abs((mk_moore_nb(sxr)+sxr)-1)
            non_env_dxs = mk_binary_domains(np.sum(sxr_ne_base).astype(int))
            for nei in non_env_dxs:
                sxr_ne = sxr_ne_base+sxr
                sxr_ne[sxr_ne_base.nonzero()] = nei
                vxs.append(sxr_ne)
            # same for transposed config
            sxrt_ne_base = abs((mk_moore_nb(sxrt)+sxrt)-1)
            non_env_dxs_sxrt = mk_binary_domains(np.sum(sxrt_ne_base).astype(int))
            for nei in non_env_dxs_sxrt:
                sxrt_ne = sxrt_ne_base+sxrt
                sxrt_ne[sxrt_ne_base.nonzero()] = nei
                vxs.append(sxrt_ne)
    # reduce list        
    for vxi in vxs:
        vx_in = False
        for sxi in sxs:
            if np.array_equal(vxi,sxi):
                vx_in = True
                break
        if not vx_in:
            sxs.append(vxi.astype(int))
    return sxs

# check activation continuity between transitions
def check_act_ct(dxs,sx,ids=True):
    if len(dxs.shape[1:]) < len(sx.shape):
        sx = sx.flatten()
    nz_ids = sum_nonzero(dxs*sx)
    if ids:
        return dxs[nz_ids],nz_ids
    return dxs[nz_ids]

# old version
# def print_ac_cases(doms,rl=0,rh=0,nonzero=True,title=''):
#     dxs = doms*1
#     # for tensors
#     if len(dxs.shape) == 3:
#         a,b,c = dxs.shape
#         dxs = dxs.reshape(a,b*c)
#     rl,rh = (rl,rh) if rh<rh else (0,dxs.shape[1])
#     nz = 0 if nonzero==True else -1
#     ids = [(ac,len(sum_is(dxs,ac))) for ac in range(rl,rh+1) if sum_is(dxs,ac).shape[0]>nz]
#     print()
#     print(title)
#     for ac,ncases in ids:
#         print('acs: {}, cases: {}'.format(ac,ncases))
#     total = sum([nc for ac,nc in ids])
#     print('total: {}'.format(total))
#     print()
# print number of cases for each n active cells in domains
def print_ac_cases(doms,rl=0,rh=0,nonzero=True,title=''):
    nz = 0 if nonzero==True else -1
    if len(doms.shape) == 2:
        max_sum = np.max(doms.sum(axis=(1))).astype(int)
    else:
        max_sum = np.max(doms.sum(axis=(1,2))).astype(int)
    rl,rh = (rl,rh) if rh<rh else (nz,max_sum)
    total = 0
    print()
    print(title)
    for ac in range(max_sum+1):
        ncases = sum_is(doms,ac).shape[0]
        if ncases > nz:
            total += ncases
            print(f'acs: {ac}, cases: {ncases}')
    print(f'total: {total}\n')

# remove isolated cells
# basically same as gol rule, but nb==0 -> cell=0
def rm_isol(dx):
    dx2 = dx*1
    for ei,vi in enumerate(dx):
        for ej,vij in enumerate(vi):
            nb = np.sum(dx[max(0,ei-1):ei+2,max(ej-1,0):ej+2]) - vij
            vx = 1 if vij==1 and nb > 0 else 0 
            dx2[ei,ej] = vx
    return dx2

# remove all zero rows/cols (assumes sx is centered in dx)
# if squared: remove considering entire row-col layers 
def rm_zero_layers(xdx,squared=False):
    dx = xdx*1
    if squared and np.sum(xdx)>0:
        while squared==False:
            dx = dx[1:-1,1:-1]
            if sum_borders(dx) > 0:
                return dx
    vs = np.sum(dx,axis=1).nonzero()[0]
    dx = dx[min(vs):max(vs)+1]
    hs = np.sum(dx,axis=0).nonzero()[0]
    dx = dx[:,min(hs):max(hs)+1]
    return dx

# adjust domains to bigger, for each dimension
def adjust_domains(x1,x2):
    # n rows/cols to fill     
    rx,cx = np.abs(np.array(x1.shape)-np.array(x2.shape))
    r0,rn = [int(rx/2)]*2 if rx%2==0 else [int(rx/2),int(rx/2)+1]
    c0,cn = [int(cx/2)]*2 if cx%2==0 else [int(cx/2),int(cx/2)+1]
    # if one is bigger than the other
    if x1.shape[0] >= x2.shape[0] and x1.shape[1] >= x2.shape[1]:
        return x1,expand_domain(x2,rows=(r0,rn),cols=(c0,cn))
    if x2.shape[0] >= x1.shape[0] and x2.shape[1] >= x1.shape[1]:
        return expand_domain(x1,rows=(r0,rn),cols=(c0,cn)),x2
    # one is bigger in rows, the other in cols
    if x1.shape[0] > x2.shape[0]:
        x2 = expand_domain(x2,rows=(r0,rn))
    elif x1.shape[0] < x2.shape[0]:
        x1 = expand_domain(x1,rows=(r0,rn))
    if x1.shape[1] > x2.shape[1]:
        x2 = expand_domain(x2,cols=(c0,cn))
    elif x1.shape[1] < x2.shape[1]:
        x1 = expand_domain(x1,cols=(c0,cn))
    return x1,x2

# pass set of arrays into tensor of gol domains for visualization
# dxs: matrix with arrays of gol sts
# sx form for reshaping
def mk_dxs_tensor(dxs,sx):
    if len(dxs.shape)==3:
        return dxs
    if dxs.shape[1] > sx.flatten().shape[0]:
        sx = expand_domain(sx)
    sxi,sxj = sx.shape
    tgol = np.zeros((dxs.shape[0],sxi,sxj))
    for di,dx in enumerate(dxs):
        tgol[di] = dx.reshape(sxi,sxj)
    return tgol
# center active cells in domain
def center_sx_in_dx(sx):
    if np.array_equal(sx,rm_zero_layers(sx)):
        return sx
    csx = expand_domain(rm_zero_layers(sx))
    if csx.shape == sx.shape:
        return csx
    if sx.shape[0] < csx.shape[0]:
        csx = csx[1:sx.shape[0]+1]
    if sx.shape[1] < csx.shape[1]:
        csx = csx[:,1:sx.shape[1]+1]
    if csx.shape == sx.shape:
        return csx
    csx,sx = adjust_domains(csx,sx)
    return csx
def center_tensor_sxs(tensor_sxs):
    for sxi,sx in enumerate(tensor_sxs):
        tensor_sxs[sxi] = center_sx_in_dx(sx)
    return tensor_sxs

# array 2 int, for domains
def mk_binary_index(px,arrays=True,dims=4):
    px = mk_min_sqrd_domain(px,moore_nb=False)
    if px.shape[0]<dims:
        px = center_sx_in_dx(np.pad(px,((0,dims-px.shape[0]),(0,0))))
    if px.shape[1]<dims:
        px = center_sx_in_dx(np.pad(px,((0,0),(0,dims-px.shape[1]))))
    px_id = array2int(px)
    if not arrays:
        return px_id
    return px_id,px

def are_the_same_sx(sx1,sx2):
    if np.sum(sx1) != np.sum(sx2):
        return False
    vxs = mk_min_sx_variants(sx1)
    sx2 = rm_zero_layers(sx2)
    for vx in vxs:
        if np.array_equal(vx,sx2):
            return True
    return False

def is_sx_in_dx(sx,dx,mk_variants=False):
    # trivial check (sx[0] of variants has env=0)
    if type(sx)==np.ndarray and np.sum(sx) > np.sum(dx):
        return False
    if type(sx)==list and np.sum(sx[0]) > np.sum(dx):
        return False
    # make variants if needed
    if mk_variants:
        vxs = mk_sx_variants(sx) 
    else:
        vxs = sx if type(sx)==list else [sx]
    # match using sliding window
    for vx in vxs:
        for wi in range(dx.shape[0]-vx.shape[0]+1):
            for wj in range(dx.shape[1]-vx.shape[1]+1):
                dw = dx[wi:wi+vx.shape[0],wj:wj+vx.shape[1]]
                if np.array_equal(dw,vx):
                    return True
    return False
# same but for the whole array of domains simultaneously
# dxs: matrix of domain-arrays, sx in matrix form
# px has to have the dimensions for reshaping dxs
# membrane = True: symsets, if False: minsets
# search_borders: to include search where maybe borders=membranes
def is_sx_in_dxs(sx,dxs,px,membrane=False,search_borders=False,tensor=False,print_data=False):
    # if membrane and px.flatten().shape[0] < 37:
    if membrane and search_borders:
        if len(dxs.shape) == 2:
            dxs = dxs.reshape(dxs.shape[0],px.shape[0],px.shape[1])
        dxs = np.pad(dxs,((0,0),(1,1),(1,1)))
        px = expand_domain(px)
    if len(dxs.shape)==3:
        dxs = dxs.reshape(dxs.shape[0],px.flatten().shape[0])
    nz_ids = np.zeros(dxs.shape[0]).astype(int)
    if membrane:
        sx = expand_domain(rm_zero_layers(sx))
        vxs = mk_sx_variants(sx,mk_non_env=False)
    else:
        vxs = mk_min_sx_variants(sx,moore_nb=False)
    for vx in vxs:
        vx_nz_ids = np.zeros(dxs.shape[0]).astype(int)
        for wi in range(px.shape[0]-vx.shape[0]+1):
            for wj in range(px.shape[1]-vx.shape[1]+1):
                wx = np.zeros((px.shape))
                wx[wi:wi+vx.shape[0],wj:wj+vx.shape[1]] = vx
                # vx is there, env is unknown
                vx_nz_ids[sum_is(dxs*wx.flatten(),np.sum(vx))] = 1
                # remove if env/memb is non zero
                if membrane:
                    vx_nz_ids[sum_nonzero(dxs*mk_moore_nb(wx).flatten())] = 0
                nz_ids += vx_nz_ids
    nz_ids = nz_ids.nonzero()[0]
    if print_data:
        print_ac_cases(dxs,title='sx in domains:')
    if tensor:
        dxs = dxs[nz_ids]
        dxs = mk_dxs_tensor(dxs,px)
        if membrane and px.flatten() < 37:
            dxs = dxs[:,1:-1,1:-1]
        return dxs,nz_ids
    return nz_ids
def is_sx_in_next(sx,dx,mk_variants=False):
    dy = gol_step(dx,expanded=True)
    return is_sx_in_dx(sx,dy,mk_variants=mk_variants)


# slow but works
# dxs: tensor of dxs 
def filter_composed_dxs(dxs,ids=False):
    for di,dx in enumerate(dxs):
        if is_composed(dx):
            dxs[di]=0
    nc_ids = sum_nonzero(dxs)
    dxs = dxs[nc_ids]
    if ids:
        return dxs,nc_ids
    return dxs
def is_composed(dx):
    ijs = [[i,j] for i,j in zip(dx.nonzero()[0],dx.nonzero()[1])]
    uxs = []
    for e1,x1 in enumerate(ijs):
        u = []
        for e2,x2 in enumerate(ijs):
            if abs(x1[0]-x2[0])>= 2 or abs(x1[1]-x2[1])>=2:
                pass
            else:
                u.extend([e1,e2])
        uxs.append(list(set(u)))
    mu = [0]
    for _ in range(3):
        for i in range(len(uxs)):
            if i in mu and i in uxs[i]:
                mu.extend(uxs[i])
    mu = list(set(mu))
    if len(mu) < len(ijs):
        return True
    return False



'''
Information related
'''
# distance matrices for intrinsic info
# for every x and y value of a,b,...,n elements: sqrt( (ax-bx)**2 + (ay-by)**2 )
# basically the euclidian distance for every comparison
def make_dms(count,whole=False):
    # transition matrix for x
    # given x, what are the probs for y
    # every value divided by the sum of the rows (values for x)
    tm_x = count/np.sum(count,axis=1)
    if whole:
        tm_x = count*1
    # transition matrix for y
    # given y, the probs of x
    # knowing y, it is each value divided by the vertical sum (values of y)
    # then transposed, so it is in function of y->x instead of x->y
    tm_y = (count/np.sum(count,axis=0)).T
    if whole:
        tm_y = np.ascontiguousarray(count.T)
    # distance matrices
    dim = tm_x.shape[0]
    # fill x
    dmx = np.zeros((dim,dim))
    for ei,i in enumerate(tm_x):
        for ej,j in enumerate(tm_x):
            dmx[ei,ej] = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    # fill y 
    dmy = np.zeros((dim,dim))
    for ei,i in enumerate(tm_y):
        for ej,j in enumerate(tm_y):
            dmy[ei,ej] = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    return dmx,dmy

'''save, load, etc'''
# def save_as(file,name,ext=''):
#     import pickle
#     import os
#     if not ext:
#         fname = name if '.' in name else '{}.{}'.format(name,'unk')
#     else:
#         fname = '{}.{}'.format(name,ext)
#     while os.path.isfile(fname):
#         i = 1
#         if len(fname.split('.')) > 2:
#             name = ''.join(fname.split('.')[:-1])
#             ext = fname.split('.')[-1]
#         else:
#             name,ext = fname.split('.')
#         try:
#             fi = int(name[-1])+1
#             fname = '{}{}.{}'.format(fname[:-1],fi,ext)
#         except:
#             fi = i+1
#             fname = '{}{}.{}'.format(fname,i,ext)
#     with open(fname,'wb') as f:
#         pickle.dump(file,f)
#     print('\nsaved as: {}\n'.format(fname))

# def load_data(filename='',ext='',dirname='gol_exps_data'):
#     import pickle
#     import os
#     dirpath = os.path.abspath(os.path.join(os.getcwd(),'..',dirname)) if dirname else os.getcwd()
#     if filename:
#         if ext:
#             filename += f'.{ext}'
#         fpath = os.path.join(dirpath,filename)
#         try:
#             with open(fpath,'rb') as fname:
#                 fdata = pickle.load(fname)
#                 return fdata
#         except:
#             print('\n{} not as path {}\n'.format(filename,fpath))
#     fnames = [i for i in os.listdir(dirpath) if '.{}'.format(ext) in i]
#     while True:
#         print()
#         for ei,fi in enumerate(fnames):
#             print('{} - {}'.format(ei+1,fi))
#         print()
#         x = input('\nfile: _ ')
#         if x == 'q' or x == 'quit':
#             return
#         try:
#             with open(fnames[int(x)-1],'rb') as fname:
#                 fdata = pickle.load(fname)
#                 return fdata
#         except:
#             print('\ninvalid input? (q/quit to quit)\n')

'''
old fxs
'''
# check for cases where sx appears in a different env
# for the basic cases: sx,e0 <-> sx,ex
# x1: the basic/known instance, to compare against
def are_sx_instances(dx1,dx2):
    x1,x2 = dx1*1,dx2*1
    x2 = rm_isol(x2)
    if np.sum(x2) > np.sum(x1):
        x2 = rm_env(x2)
    if np.sum(x1) == np.sum(x2):
        return are_symmetrical(x1,x2)
    return False
# check symmetries in 2 gol domains 
# x1,x2: matrix form gol reps
def are_symmetrical(x1,x2,nrolls=0):
    # if sizes don't match, adjust to the larger one
    if x1.shape != x2.shape:
        x1,x2 = adjust_domains(x1,x2)
    # if not specified, assume all
    nrolls = x1.flatten().shape[0] if nrolls==0 else nrolls
    # rotations
    for ri in range(4):
        # rotations
        x2r = np.rot90(x2,ri)
        if np.array_equal(x1,x2r):
            return True
        # transpositions
        x2rt = np.ascontiguousarray(x2r.T)
        if np.array_equal(x1,x2rt):
            return True
        # translations
        if check_translation(x1,x2r,x2rt):
            return True
    return False
def is_block(domx,e0=False):
    # no more than block + m0 + full env
    # min m0 = 2 sides = 5 cells
    n,m = domx.shape
    if 4 <= np.sum(domx) <= n*m-5:
        for i in range(n):
            for j in range(m):
                if np.sum(domx[i:i+2,j:j+2]) == 4:
                    if np.sum(domx) == 4:
                        return True
                    if not e0:
                        if np.sum(domx[max(0,i-1):i+3,max(0,j-1):j+3]) == 4:
                            return True
    return False
def is_blinker(domx):
    if np.sum(domx) == 3:
        vsum = np.sum(domx,axis=0)
        hsum = np.sum(domx,axis=1)
        if 3 in vsum or 3 in hsum:
            return True
    return False
def is_protoblock0(domx,pbx,pbx_type):
    m,n = domx.shape
    # def protoblock and min 'membrane'
    if pbx_type == 'pb0':
        pb_n0 = 6 # 5 for 2 sides + 1 interior
    elif pbx_type == 'pb2':
        pb = np.zeros((2,4))
        pb[1,:2] = 1
        pb[0,2:] = 1
        pb_n0 = 11 # 7 for 2 sides + 4 interiors
    if np.sum(pbx) <= np.sum(domx) <= m*n - pb_n0:
        if pbx_type=='pb0':
            for i in range(m):
                for j in range(n):
                    if np.sum(domx[i:i+2,j:j+2]) == 3 and np.sum(domx[max(0,i-1):i+3,max(0,j-1):j+3]) == 3:
                        return True
        else:
            pass
    return False
def is_block_next(domx,e0=False,expanded=True):
    # block may be outside of current domain
    domy = gol_step(domx,expanded=expanded)
    return is_block(domy,e0)

# expand domain for rolling correctly
# basically they should mantain the dist among act cells
def check_translation(x1,x2r,x2t):        
    pb = reduce_to_min_sqr(pb)
    n,m = x1.shape
    b1 = np.zeros((n+2,m+2))
    b1[1:-1,1:-1] = x1
    bx1 = b1.flatten().nonzero()[0]
    bx1 = np.abs(bx1-bx1[0])
    b2r = np.zeros((n+2,m+2))
    b2r[1:-1,1:-1] = x2r
    bx2r = b2r.flatten().nonzero()[0]
    if np.array_equal(bx1,np.abs(bx2r-bx2r[0])):
        return True
    if np.array_equal(bx1,np.abs(np.flip(bx2r-bx2r[0]))):
        return True
    b2t = np.zeros((n+2,m+2))
    b2t[1:-1,1:-1] = x2t
    bx2t = b2t.flatten().nonzero()[0]
    if np.array_equal(bx1,np.abs(bx2t-bx2t[0])):
        return True
    if np.array_equal(bx1,np.abs(np.flip(bx2t-bx2t[0]))):
        return True
    return False

# split using rows/cols=0
def rm_env(dx,nc=2):
    dom = dx*1
    # first/last row/col dont change results
    r0 = sum_is(dom[1:-1],0)+1
    c0 = sum_is(dom[:,1:-1],0,axis=0)+1
    i0,j0 = 0,0
    # basically split by rows/cols=0
    if np.sum(r0)>0:
        for i in list(r0)+[dom.shape[0]]:
            if np.sum(dom[i0:i]) <= nc:
                dom[i0:i] = 0
                i0 = i
            else:
                c0 = sum_is(dom[i0:i],0,axis=0)
                for j in list(c0)+[dom.shape[0]]:
                    if np.sum(dom[i0:i,j0:j]) <= nc:
                        dom[i0:i,j0:j] = 0
                    j0 = j
            i0,j0 = i,0
        return dom
    for j in list(c0)+[dom.shape[1]]:
        if np.sum(dom[:,j0:j]) <= nc:
            dom[:,j0:j] = 0
            j0 = j
        else:
            r0 = sum_is(dom[:,j0:j],0)
            for i in list(r0)+[dom.shape[1]]:
                if np.sum(dom[i0:i,j0:j]) <= nc:
                    dom[i0:i,j0:j] = 0
                    i0 = i
        j0,i0 = j,0
    return dom

def reduce_to_min_sqr(xa,xb):
    x1,x2 = xa*1,xb*1
    x1 = expand_domain(rm_zero_layers(x1))
    x2 = expand_domain(rm_zero_layers(x2))
    if x1.shape == x2.shape and x1.shape[0] == x1.shape[1]:
        return xa,xb
    rx,cx = np.array(x2.shape) - np.array(x1.shape)
    if rx > 0:
        x1 = np.pad(x1,((0,rx),(0,0)))
    elif rx < 0:
        x2 = np.pad(x2,((0,abs(rx)),(0,0)))
    if cx > 0:
        x1 = np.pad(x1,((0,0),(0,cx)))
    elif cx < 0:
        x2 = np.pad(x2,((0,0),(0,abs(cx))))
    if x1.shape[0] == x1.shape[1]:
        return x1,x2
    if x1.shape[0]-x1.shape[1] > 0:
        return np.pad(x1,((0,0),(0,x1.shape[0]-x1.shape[1]))), np.pad(x2,((0,0),(0,x1.shape[0]-x1.shape[1])))
    return np.pad(x1,((0,x1.shape[1]-x1.shape[0]),(0,0))), np.pad(x2,((0,x1.shape[1]-x1.shape[0]),(0,0)))

def get_diagonals(dx,min_len=1,sums=False,seps=False):
    if seps:
        diagonals_lr = []
        diagonals_rl = []
    diagonals = []
    for di in range(-dx.shape[0]+1,dx.shape[1]):
        diag_lr = dx.diagonal(di)
        diag_rl = np.fliplr(dx).diagonal(di)
        if len(diag_lr) >= min_len:
            if sums:
                if seps:
                    diagonals_lr.append(np.sum(diag_lr))
                    diagonals_rl.append(np.sum(diag_rl))
                else:
                    diagonals.extend([np.sum(diag_lr),np.sum(diag_rl)])
            else:
                if seps:
                    diagonals_lr.append(diag_lr)
                    diagonals_rl.append(diag_rl)
                else:
                    diagonals.extend([diag_lr,diag_rl])
    if seps:
        return diagonals_lr,diagonals_rl
    return diagonals

# check if some instance (sx) can be found in a domain (dx)
def is_in_domain(sx,dx,zeros=False):
    # reduced to the smallest squared shape possible (sqr for T)
    sx,dx = reduce_to_min_sqr(sx,dx)
    m,n = dx.shape
    # check in domain
    for ri in range(4):
        sxr = np.rot90(sx,ri)
        for t in range(m*n):
            if zeros==True:
                sxr = mk_moore_nb(sxr)+sxr
                if dx.shape != sxr.shape:
                    import pdb; pdb.set_trace()
                if np.sum(dx*np.roll(sxr,t))==np.sum(sx):
                    if are_symmetrical(sxr,dx*np.roll(sxr,t)):
                        return True
            elif not zeros and np.sum(dx*np.roll(sxr,t))>0:
                if are_symmetrical(sxr,dx*np.roll(sxr,t)):
                    return True
        sxrt = np.ascontiguousarray(sxr.T)
        for t in range(m*n):
            if zeros==True:
                sxrt = mk_moore_nb(sxrt)+sxrt
                if np.sum(dx*np.roll(sxrt,t))==np.sum(sx):
                    if are_symmetrical(sxrt,dx*np.roll(sxrt,t)):
                        return True
            elif not zeros and np.sum(dx*np.roll(sxrt,t))>0:
                if are_symmetrical(sxrt,dx*np.roll(sxrt,t)):
                    return True
    return False
# just to get ids as from the whole proto domain
# dxs_ids = array of ids
# sms_cases = matrix of arrays: [ac,idx,idy]
def match_dxs_sms_ids(dxs_ids,sms_cases):
    cases = np.pad(sms_cases.astype(int),(0,2))*1
    cases[:,3] = dxs_ids[cases[:,1]]
    cases[:,4] = dxs_ids[cases[:,2]]
    return cases
# remove basic patterns present in domains and check symmetries again
# dxs: e=tensor of matrix shaped domains
# dx_div: to delimit basic patterns (inclusive) range and dx of search (higher than)
def check_basic_patterns(dxs,sx=[],dx_div=4,ids=False,print_data=True):
    # in case dxs is not a tensor
    tensor = True
    if len(dxs.shape) == 2:
        dxs = mk_dxs_tensor(dxs,sx)
        tensor = False
    # this is easier by hand
    bp1 = expand_domain(np.ones((1,2)))
    bp2 = expand_domain(np.array([1,0,0,1]).reshape(2,2))
    ids_bp = []
    for bpi in [bp1,bp1.T,bp2,np.rot90(bp2,1)]:
        for di in sum_higher(dxs,dx_div):
            dx = expand_domain(dxs[di])
            for wi in range(dx.shape[0]-bpi.shape[0]+1):
                for wj in range(dx.shape[1]-bpi.shape[1]+1):
                    dw = dx[wi:wi+bpi.shape[0],wj:wj+bpi.shape[1]]
                    if np.array_equal(dw,bpi):
                        ids_bp.append(di)
    if len(ids_bp)>0:
        dxs[np.array(ids_bp)] = 0
    non_bp_ids = sum_nonzero(dxs)
    dxs = dxs[non_bp_ids]
    if dx_div < 3:
        if print_data:
            print_ac_cases(dxs,title='after filtering basic comps <= {}:'.format(dx_div))
        if ids:
            return dxs,non_bp_ids
        return dxs
    # translation, transopsition, rotation, different non moore envs
    ids_bpxs,bxs = [],[]
    print()
    for bi in tqdm(sum_in_range(dxs,3,dx_div+1).astype(int)):
        bx = expand_domain(rm_zero_layers(dxs[bi]))
        for ri in range(4):
            br = np.rot90(bx,ri)
            bms = []
            bmx = abs((mk_moore_nb(br)+br)-1)
            non_moore_dxs = mk_binary_domains(np.sum(bmx).astype(int))
            for mi in non_moore_dxs:
                bm = bmx+br
                bm[bmx.nonzero()] = mi
                bms.append(bm)
            for bh in [br,br.T]+bms:
                bh_in = False
                for bxi in bxs:
                    if np.array_equal(bh,bxi):
                        bh_in = True
                        break
                if not bh_in:
                    bxs.append(bh)
    # search in domains
    for bi in tqdm(range(len(bxs))):
        bx = bxs[bi]
        for di in sum_higher(dxs,dx_div):
            dx = expand_domain(dxs[di])
            for wi in range(dx.shape[0]-bx.shape[0]+1):
                for wj in range(dx.shape[1]-bx.shape[1]+1):
                    dw = dx[wi:wi+bx.shape[0],wj:wj+bx.shape[1]]
                    if np.array_equal(dw,bx):
                        ids_bpxs.append(di)
    # all_ids = ids_bp + ids_bpxs
    if len(ids_bpxs)==0:
        if ids:
            return dxs,sum_nonzero(dxs)
        return dxs
    dxs[np.array(ids_bpxs)] = 0
    dxs = sum_nonzero(dxs,arrays=True)
    if not tensor:
        dxs = dxs.reshape(dxs.shape[0],sx.flatten().shape[0])
    if print_data:
        print_ac_cases(dxs,title='after filtering basic comps instances <= {}:'.format(dx_div))
    if ids:
        dxs_ids = sum_nonzero(dxs)
        return dxs,dxs_ids
    return dxs
'''
end old fxs
'''


'''
analysis and visualization fxs
'''

# directed graph
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


def mk_ox_graph(ox,min_ny=0,closed=False,layout=0):
    if min_ny>0:
        rm = []
        for node in ox.nodes:
            c = 0
            for edge in ox.edges:
                if node==edge[1]:
                    c += 1
            if c<min_ny:
                rm.append(node)
        ox.remove_nodes_from(rm)
    if closed:
        ex = [e[0] for e in ox.edges]
        ey = [e[1] for e in ox.edges]
        nc = [ni for ni in ox.nodes if ni in ex and ni in ey]
        rm = [nr for nr in ox.nodes if nr not in nc]
        ox.remove_nodes_from(rm)
        label_dict = {}
        for nid,label in ox.nodes.data("label"):
            label_dict[nid] = label
    edge_colors = [i[2] for i in ox.edges.data("weight")]
    node_sizes = []
    for node in ox.nodes:
        node_sizes.append(np.where(np.array([i[1] for i in ox.edges])==node)[0].shape[0])
    if closed:
        node_sizes = [1000*ns for ns in node_sizes]
        # node_sizes = [0 for ns in node_sizes]
    if layout==0:
        pos = nx.circular_layout(ox)
    else:
        pos = nx.spring_layout(ox)
    cmap = plt.cm.plasma
    if closed:
        nodes = nx.draw_networkx_nodes(ox,pos,node_size=node_sizes,node_color="w",edgecolors='k',alpha=0.7)
        edges = nx.draw_networkx_edges(ox,pos,arrowstyle="->",arrowsize=10,edge_color=edge_colors,edge_cmap=cmap,width=2)
    else:
        nodes = nx.draw_networkx_nodes(ox,pos,node_size=node_sizes,node_color="indigo")
        edges = nx.draw_networkx_edges(ox,pos,arrowstyle="->",arrowsize=10,edge_color=edge_colors,edge_cmap=cmap,width=0.5)
    if closed:
        labels = nx.draw_networkx_labels(ox,pos,label_dict,font_size=8,alpha=1)
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    ax = plt.gca()
    ax.set_axis_off()
    # ax.figure.set_size_inches(10,10)
    plt.colorbar(pc, ax=ax)
    plt.tight_layout()
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.show()

def mk_env_data_plots(dix):
    for dk,dv in dix.items():
        if dk>0:
            envs = dv['envs']
            dx_envs = dix[0]['sx_dx']*-1 + mk_moore_nb(dix[0]['sx_dx'])*np.max(envs)*2
            dx_envs += envs
            pdist = []

def mk_env_distinctions(sx_dx,sx_dxs,sx_ids,sx_name='',mk_plots=False):
    from scipy.stats import wasserstein_distance as wdist
    env_info = {}
    for key,val in sx_ids.items():
        ncases = sx_ids[key].shape[0]
        # all envs overlapped
        env = np.sum(sx_dxs[val],axis=0).reshape(sx_dx.shape) * np.abs(sx_dx-1)
        edx = sx_dx*-1 + mk_moore_nb(sx_dx)*np.max(env)*2
        edx += env
        # interactional domain
        idx = mk_moore_nb(mk_moore_nb(sx_dx)) - sx_dx
        # dists and plots
        idx_counts = edx[np.where(idx==1)]
        idx_dist = idx_counts/np.sum(idx_counts)
        # emdx = emd_samples(idx_dist,np.ones(idx_dist.shape)/idx_dist.shape)
        emdx = emd_samples(idx_dist,np.ones(idx_dist.shape)/idx_dist.shape)     # same vals
        env_info[key] = [emdx,ncases,env,edx,idx_counts,idx_dist]
    eks = sorted([[key,val[0],val[1]] for key,val in env_info.items()],key=lambda x:x[1],reverse=True)
    # plots
    while mk_plots==True:
        print()
        for ei,(key,emdx,ncases) in enumerate(eks):
            print('[{}] - dxid:{}, emd = {}, ncases = {}'.format(ei,key,emdx,ncases))
        kx = input('\n? _ ')
        if kx == 'q' or kx == 'quit':
            mk_plots = False
        else:
            try:
                key,emd_val,ncases = eks[int(kx)]
                print(key,emd_val,ncases)
                emdx,ncases,env,edx,idx_counts,idx_dist = env_info[key]
                # plots
                plt.plot(idx_counts)
                plt.show()
                cmap = mpl.cm.get_cmap("jet").copy()
                imx = plt.imshow(edx, vmin=0, vmax=np.max(env), cmap=cmap, aspect='auto')
                imx.cmap.set_over('white')
                imx.cmap.set_under('black')
                plt.colorbar()
                plt.title('{} environmental distinctions\nEMD info = {}'.format(sx_name,round(emdx,3)))
                plt.show()
            except:
                print('\nunknown input?\n')
    return env_info

def plot_patterns(ufs,rows=3,cols=4,size=(9,6)):
    fig,axs = plt.subplots(nrows=rows,ncols=cols,
                           figsize=size)
                           # subplot_kw={'xticks':[],'yticks':[]})
    # plt.axis('off')
    palette = np.array([[255, 255, 255],   # 0:white
                        [  0, 255,   0],   # 1:green
                        [  0,   0, 255],   # 2:blue
                        [255,   0,   0],   # 3:red
                        [  0,   0,   0]])  # 4:black
    for ax,uf in zip(axs.flat,ufs):
        sx = uf[0]['sx']
        dx = sx + mk_moore_nb(sx)*2
        ax.imshow(palette[dx.astype(int)])#,cmap='binary')
        # ax.grid(visible=True)
        ax.set_title(uf[0]['label'])
        ax.axis('off')
    plt.tight_layout
    plt.show()




# def mk_sym_sx_ids(symsets):
#     sym_ids = {}
#     for sym in symsets:
#         sym = rm_zero_layers(syml)
#         sym_id = array2int(sym)
#         sym_ids[sym_id] = sym
#     return sym_ids

