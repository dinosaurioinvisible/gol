
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pdb

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
            return matrix[np.where(np.sum(matrix,axis=(1,2))>x)[0]]
        return np.where(np.sum(matrix,axis=(1,2))>x)[0]
    if arrays:
        return matrix[np.where(np.sum(matrix,axis=axis)>x)[0]]
    return np.where(np.sum(matrix,axis=axis)>x)[0]
def sum_lower(matrix,x,axis=1,arrays=False):
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

# to check if some sx is surrended by zeros in some domx 
def sum_borders(domx):
    cx = np.sum(domx[1:-1,1:-1])
    return np.sum(domx)-cx
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

# make canonical gol patterns (sx,e=0) from word inputs
# minimal form: active cells + moore neighborhood rectangled
def mk_gol_pattern(px):
    if px == 'block':
        dx = np.zeros((4,4))
        dx[1:-1,1:-1] = 1
        return dx
    elif px == 'pb0':
        dx = np.ones((2,2))
        dx[0,1] = 0
        dx = np.pad(dx,(1,1))
        return dx
    elif px == 'pb2':
        dx = np.zeros((2,4))
        dx[1,:2] = 1
        dx[0,2:] = 1
        dx = np.pad(dx,(1,1))
        return dx
    elif px == 'blinker':
        d1 = np.zeros((5,3))
        d1[1:-1,1] = 1
        d2 = np.ascontiguousarray(d1)
        dx = [d1,d2]
        return dx
    elif px == 'glider':
        dx = []
        d1 = np.zeros((5,5))
        d1[1,2] = 1
        d1[2,3] = 1
        d1[3,1:-1] = 1
        d2 = np.zeros((5,5))
        d2[1,1] = 1
        d2[2:-1,2] = 1
        d2[1:3,3] = 1
        for di in [d1,d2]:
            for ri in range(4):
                dr = np.rot90(di,ri)
                dt = np.ascontiguousarray(dr.T)
                dx.extend([dr,dt])
        return dx
    else:
        print('\npattern not defined\n')

# pass set of arrays into tensor of gol domains for visualization
# dxs: matrix with arrays of gol sts
# sx form for reshaping
def mk_dxs_tensor(dxs,sx):
<<<<<<< HEAD
    if dxs.shape[1] != sx.flatten().shape[0]:
=======
    if len(dxs.shape)==3:
        return dxs
    if dxs.shape[1] > sx.flatten().shape[0]:
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
        sx = expand_domain(sx)
    sxi,sxj = sx.shape
    tgol = np.zeros((dxs.shape[0],sxi,sxj))
    for di,dx in enumerate(dxs):
        tgol[di] = dx.reshape(sxi,sxj)
    return tgol

def sort_by_sum(dxs):
    return np.array(sorted(list(dxs),key=lambda x:np.sum(x)))

def center_tensor_sxs(tensor_sxs):
    for sxi,sx in enumerate(tensor_sxs):
        tensor_sxs[sxi] = center_sx_in_dx(sx)
    return tensor_sxs
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
<<<<<<< HEAD
    if sx.shape[0]*sx.shape[1] > 30 and expanded == True:
        for di in tqdm(range(sx_domains.shape[0])):
            dx = sx_domains[di].reshape(sx.shape)
            if np.sum(dx)>2:
                sxys[di] = gol_step(dx,expanded=True).flatten()
=======
    if sx_domains.shape[0] > 2**12:
        for di in tqdm(range(sx_domains.shape[0])):
            dx = sx_domains[di].reshape(sx.shape)
            if np.sum(dx)>2:
                sxys[di] = gol_step(dx,expanded=expanded).flatten()
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
        return sxys
    # simulate transitions
    for di,dx in enumerate(sx_domains):
        if np.sum(dx)>2:
            sxys[di] = gol_step(dx.reshape(sx.shape),expanded=expanded).flatten()
    if mk_zero:
        # remove decaying domains (<3)
        sxys[sum_lower(sxys,3)] = 0
    return sxys

# remove decaying activations within domain (not environment of sx)
# dxs: domains in matrix form
# sx: pattern being analyzed, but only for reshaping
<<<<<<< HEAD
def rm_non_env(dxs,sx,print_data=True):
    # if extended dx
    if sx.flatten().shape[0] < dxs.shape[1]:
        sx = expand_domain(sx)
    dxs_ne = dxs*1
    for di,dx in enumerate(dxs):
        gx = gol_step(dx.reshape(sx.shape))
        dxs_ne[di] = ((mk_moore_nb(gx)+gx)*dx.reshape(sx.shape)).flatten()
    if print_data:
        print_ac_cases(dxs_ne,title='after removing decaying non-env cells:')
=======
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
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    return dxs_ne

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

# check for patterns 
# domx: gol domain/lattice in matrix form
# e0: empty environment
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
<<<<<<< HEAD
def is_protoblock(domx,pbx,pbx_type):
=======
def is_protoblock0(domx,pbx,pbx_type):
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
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
<<<<<<< HEAD
                    
=======
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
# same for next timestep
def is_block_next(domx,e0=False,expanded=True):
    # block may be outside of current domain
    domy = gol_step(domx,expanded=expanded)
    return is_block(domy,e0)
# for proto blocks
def is_pb_next(domx,pbx,pb_type,expanded=True):
    domy = gol_step(domx,expanded=expanded)
<<<<<<< HEAD
    return is_protoblock(domy,pbx,pb_type)
=======
    return is_protoblock0(domy,pbx,pb_type)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
# blinker
def is_blinker(domx):
    if np.sum(domx) == 3:
        vsum = np.sum(domx,axis=0)
        hsum = np.sum(domx,axis=1)
        if 3 in vsum or 3 in hsum:
            return True
    return False
# general function 
def is_sx_in_next(sx,dx,mk_variants=False):
    dy = gol_step(dx,expanded=True)
    return is_sx_in_dx(sx,dy,mk_variants=mk_variants)
def is_sx_in_dx(sx,dx,mk_variants=False):
    # trivial check (sx[0] of variants has env=0)
    if type(sx)==np.ndarray and np.sum(sx) > np.sum(dx):
        return False
    if type(sx)==list and np.sum(sx[0]) > np.sum(dx):
        return False
    # make variants if needed
    vxs = mk_sx_variants(sx) if mk_variants==True else sx
    # match using sliding window
    for vx in vxs:
        for wi in range(dx.shape[0]-vx.shape[0]+1):
            for wj in range(dx.shape[1]-vx.shape[1]+1):
                dw = dx[wi:wi+vx.shape[0],wj:wj+vx.shape[1]]
                if np.array_equal(dw,vx):
                    return True
    return False
<<<<<<< HEAD

=======
# same but for the whole array of domains simultaneously
# dxs: matrix of domain-arrays, sx in matrix form
# sx has to have the dimensions for reshaping dxs
# moore nb = True: symsets, if False: minsets
def is_sx_in_dxs(sx,dxs,moore_nb=True,tensor=False,print_data=False):
    if moore_nb and sx.flatten().shape[0] < 36:
        dxs = np.pad(dxs.reshape(dxs.shape[0],sx.shape[0],sx.shape[1]),((0,0),(1,1),(1,1)))
        sx = expand_domain(sx)
    if len(dxs.shape)==3:
        dxs = dxs.reshape(dxs.shape[0],sx.flatten().shape[0])
    nz_ids = np.zeros(dxs.shape[0]).astype(int)
    vxs = mk_min_sx_variants(sx,moore_nb=moore_nb)
    for vx in vxs:
        vx_nz_ids = np.zeros(dxs.shape[0]).astype(int)
        for wi in range(sx.shape[0]-vx.shape[0]+1):
            for wj in range(sx.shape[1]-vx.shape[1]+1):
                wx = np.zeros((sx.shape))
                wx[wi:wi+vx.shape[0],wj:wj+vx.shape[1]] = vx
                # vx is there, env is unknown
                vx_nz_ids[sum_is(dxs*wx.flatten(),np.sum(vx))] = 1
                # remove if env is non zero
                if moore_nb:
                    vx_nz_ids[sum_nonzero(dxs*mk_moore_nb(wx).flatten())] = 0
                nz_ids += vx_nz_ids
    nz_ids = nz_ids.nonzero()[0]
    if print_data:
        print_ac_cases(dxs,title='sx in domains:')
    if tensor:
        dxs = dxs[nz_ids]
        dxs = mk_dxs_tensor(dxs,sx)
        if moore_nb:
            dxs = dxs[:,1:-1,1:-1]
        return dxs,nz_ids
    return nz_ids
def are_the_same_sx(sx1,sx2):
    if np.sum(sx1) != np.sum(sx2):
        return False
    vxs = mk_min_sx_variants(sx1)
    sx2 = rm_zero_layers(sx2)
    for vx in vxs:
        if np.array_equal(vx,sx2):
            return True
    return False

# same but only rotation and transposition
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
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
# make all variants (rotation,transposition,non-env/moore nb)
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
            sxs.append(vxi)
    return sxs
        
# same, but on top of the next, specific for proto-domains of sx
def mk_proto_domains(sx):
    n_cells = sx.flatten().shape[0]
    return mk_binary_domains(n_cells)
# a tensor for all binary combinations
def mk_binary_domains(n_cells):
    n_cells = n_cells if type(n_cells)==int else int(n_cells)
    doms = np.zeros((2**n_cells,n_cells)).astype(int)
    for i in range(n_cells):
        f = 2**i
        xi = np.concatenate((np.zeros(f),np.ones(f)))
        n = int(2**n_cells/(2**(i+1)))
        doms[:,-1-i] = np.tile(xi,n)
    return doms

<<<<<<< HEAD
=======
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

>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
# more general fx, for gol patterns
# domain environmental (sx + env) tensor
# given a block, blinker or any other structure from the gol (sx)
# make all the env arrays for sx
# e_cells are all the cells in the environment
def mk_sx_domains(sx,membrane=False):
    # just not to call it apart
    if membrane:
        return mk_sx_membrane_domains(sx)
    # for unnamed patterns/domains
    if type(sx) == np.ndarray:
<<<<<<< HEAD
        sx = expand_domain(rm_zero_layers(sx)) # centering
        if sx.shape[0] + 2 == sx.shape[1]:
            sx = np.pad(sx,((1,1),(0,0)))
=======
        # centering and check squared domain 
        sx = mk_min_sqrd_domain(sx,moore_nb=True)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
        sx_env = mk_moore_nb(sx)
        binary_dxs = mk_binary_domains(np.sum(sx_env).astype(int))
        non_env_ids = np.where(sx_env.flatten()==0)[0]
        non_env_ids -= np.arange(non_env_ids.shape[0])
        binary_dxs = np.insert(binary_dxs,non_env_ids,1,axis=1)
        non_ids = np.where((sx+sx_env).flatten()==0)
        binary_dxs[:,non_ids] = 0
        return binary_dxs
    # number of env cells
    # pblock1: 3 active cells in the same region of the block
    if sx == 'block' or sx == 'pb0':
        e_cells = 12
    # all possibilities of binary domain
    doms = mk_binary_domains(e_cells)
    # insert sx: for every array, change sx cells into 1
    if sx=='block':
        doms = np.insert(doms,(5,5,7,7),1,axis=1)
    elif sx=='pb0':
        # corner cell not part of env domain (0 for simplicity)
        doms = np.insert(doms,3,0,axis=1)
        doms = np.insert(doms,(5,8,8),1,axis=1)
    return doms
# same, but with membranes (as in Beer studies)
def mk_sx_membrane_domains(sx):
    if sx == 'block' or sx == 'pb0':
        env_cells = 20
    dxs = mk_binary_domains(env_cells)
    if sx == 'block':
        # membrane & active cells
        dxs = np.insert(dxs,(7,7,7,7,9,9,9,9,11,11,11,11,13,13,13,13),0,axis=1)
        dxs[:,np.array([14,15,20,21])] = 1
    elif sx == 'pb0':
        dxs = np.insert(dxs,(5,6,6,6,9,9,9,9,11,11,11,11,13,13,13,13),0,axis=1)
        dxs[:,np.array([14,20,21])] = 1
    return dxs

# helper fx for continuity
# dxs: matrix of gol domain sts arrays 
# psx: primary pattern/structure determining ct with env=0
def apply_ct(dxs,psx,rm_zeros=True,ct_ids=False,non_ct_ids=False,print_data=False):
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

# get all sx: sx -> sy
# sy: any specific gol pattern domain (block,pb0,etc)
# requires matrix/lattice min domain input
# sy_px: name of sy expected pattern ('block','blinker',etc)
# dxs: specific (prev. filetered) domain
def get_sxs_from_sy(sy,sy_px,dxs=[],xpn=True,e0=False,ct=True,print_data=True,return_data=False):
    # array for expected sy
    n,m = sy.shape
    if len(dxs)==0:
        # all possible domains for (sx,ex)
        dxs = mk_binary_domains(n*m)
    # analyze domains transitions
    # sxs = []
    for di in tqdm(range(dxs.shape[0])):
        dx = dxs[di].reshape(n,m)
        if sy_px == 'block':
            # if is_block_next(dx.reshape(n,m),e0=e0,expanded=xpn):
            #     sxs.append(dx)
            if not is_block_next(dx,e0=e0,expanded=xpn):
                dxs[di] = 0
        elif sy_px[:2] == 'pb':
            if not is_pb_next(dx,sy,sy_px,expanded=xpn):
                #  sxs.appendd(dx)
                dxs[di] = 0
    # sxs = np.array(sxs)
    # print('\nproto domains: {}/{}'.format(sxs.shape[0],dxs.shape[0]))
    dxs_ids = sum_nonzero(dxs)
    print('\nproto domains: {}/{}'.format(dxs_ids.shape[0],dxs.shape[0]))
    dxs = dxs[dxs_ids]
    if ct:
        dxs,ct_ids = apply_ct(dxs,sy,ct_ids=True)
        print('\nafter CT: {}'.format(ct_ids.shape[0]))
    if print_data:
        print_ac_cases(dxs,title='proto domains:')
    if return_data:
        dxs_ct_ids = dxs_ids[ct_ids]
        return dxs,dxs_ids,dxs_ct_ids
    return dxs
# simpler version, to apply methods separatedly
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

# same, but simpler, to apply things separatedly
<<<<<<< HEAD
def mk_sxys_from_sx(sx,sx_dxs=[],ids=True,print_data=True):
    if len(sx_dxs)==0:
        sx_dxs = mk_sx_domains(sx)
    sxys = multi_gol_step(sx_dxs,sx,expanded=True)
=======
def mk_sxys_from_sx(sx,sx_dxs=[],ids=True,expanded=True,print_data=True):
    if len(sx_dxs)==0:
        sx_dxs = mk_sx_domains(sx)
    # expanded only makes sense if for sx->sy, if dx(sy) < moore_nb(sy)
    if sum_borders(sx)==0:
        expanded = False
    sxys = multi_gol_step(sx_dxs,sx,expanded=expanded)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    nz_ids = sum_higher(sxys,2)
    # nz_ids = sum_nonzero(sxys)
    sxys = sxys[nz_ids]
    sx_dxs = sx_dxs[nz_ids]
    if print_data:
        print_ac_cases(sxys,title='non zero sxys fwd domains:')
    if ids:
        return sx_dxs,sxys,nz_ids
    return sx_dxs,sxys
# get sys from sx
# in this case we can't assume what sy is valid or not
# so we should look for self-sustaining resulting patterns
# sx: initial gol pattern in matrix form
<<<<<<< HEAD
# dxs: tensor for all domains for sx
=======
# dxs: tensor for all domains 
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
# decay txs: number of sy->z1->z2->...->zn transitions
# mk_zero: auto make zero domains < 3
def get_sxys_from_sx(dxs,sx,mk_zero=True,expanded=True,ct=True,decay_txs=3,expanded_decay=False,decay_ct=True,print_data=True,return_data=False):
    # gol step for every array
    sxys = multi_gol_step(dxs,sx,mk_zero=mk_zero,expanded=expanded)
    if return_data:
        sxys_all = sxys*1
    nz_ids = sum_nonzero(sxys)
    sxys = sxys[nz_ids]
    if print_data:
        print('\nnon zero dxs in sy: {}/{}'.format(sxys.shape[0],sum_nonzero(dxs).shape[0]))
    dxs = dxs[nz_ids]
    sx = expand_domain(sx) if expanded else sx
    if ct:
        sxys,ct_ids = apply_ct(sxys,sx,ct_ids=True)
        dxs = dxs[ct_ids]
        if print_data:
            print('\nnon zero after CT: {}'.format(sum_nonzero(sxys).shape[0]))
    # decay txs for every array
    if decay_txs>0:
        # y -> z1 -> z2 ->...-> zn decay txs (often 2 or 3) (includes ct)
        sxys,z_ids = mk_dxs_decay(sxys,sx,decay_txs=decay_txs,expanded=expanded_decay,ct=decay_ct,print_data=print_data)
        dxs = dxs[z_ids]
    if print_data:
        print_ac_cases(sxys,title='sxys cases')
    if return_data:
        return sxys_all,sxys,nz_ids,ct_ids
    return dxs,sxys

# same, but super simple
<<<<<<< HEAD
def mk_yz_decay(sxys,sx,ids=False,print_data=True):
    yz = multi_gol_step(sxys,sx,expanded=True)
    yz_ids = sum_nonzero(yz)
=======
# sxys: domains in matrix form, sx: in matrix form
def mk_yz_decay(sxys,sx,expanded=True,ids=False,print_data=True):
    if sx.flatten().shape[0] < sxys.shape[1]:
        sx = expand_domain(sx)
    if sum_borders(sx) == 0:
        expanded = False
    yz = multi_gol_step(sxys,sx,expanded=expanded)
    yz_ids = sum_higher(yz,2)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    sxys = sxys[yz_ids]
    if print_data:
        print_ac_cases(sxys,title='after yz decay:')
    if ids:
        return sxys,yz_ids
    return sxys
# check & discard domains transition into 0
# dxs: matrix of arrays for gol domains
# sx: for reshaping gol arrays for sx -> sy txs
def mk_dxs_decay(dxs,sx,decay_txs=2,expanded=False,ct=True,z_arrays=False,print_data=False):
    dzs,psx = dxs*1,dxs*1
    # y -> z transition into future 
    for txi in range(decay_txs):
        dzs = multi_gol_step(dzs,sx,mk_zero=True,expanded=expanded)
        if ct:
            psx = psx if not expanded else expand_multiple_domains(psx,sx)
            nz,non_ct_ids = apply_ct(dzs,psx,non_ct_ids=True)
            dzs[non_ct_ids] = 0
            psx = dzs*1
        if print_data:
            title = 'non zero dzs in tx{}: {}'.format(txi+1,sum_nonzero(dzs).shape[0])
            print_ac_cases(sum_nonzero(dzs,arrays=True),title=title)
    # apply ids only at the end to avoid mismatches
    yz_ids = sum_nonzero(dzs)
    if z_arrays:
        return dxs[yz_ids],dzs[yz_ids],yz_ids
    return dxs[yz_ids],yz_ids

# gets number of cases of n active cells from domain
def get_ac_cases(dxs,ac=0,rl=0,rh=0,arrays=False,ids=False,nonzero=False):
    if ac>0:
        if arrays:
            return sum_is(dxs,ac,arrays=True)
        if ids:
            return sum_is(dxs,ac)
        return sum_is(dxs,ac).shape[0]
    rl,rh = (rl,rh) if rl<rh else (0,dxs.shape[1])
    nz = 0 if nonzero==True else -1
    ids = [sum_is(dxs,ac) for ac in range(rl,rh+1) if sum_is(dxs,ac).shape[0]>nz]
    if arrays:
        return dxs[ids]
    if ids:
        return ids
    return np.array([i.shape[0] for i in ids])
# same, only for printing
def print_ac_cases(doms,rl=0,rh=0,nonzero=True,title=''):
    dxs = doms*1
    # for tensors
    if len(dxs.shape) == 3:
        a,b,c = dxs.shape
        dxs = dxs.reshape(a,b*c)
    rl,rh = (rl,rh) if rh<rh else (0,dxs.shape[1])
    nz = 0 if nonzero==True else -1
    ids = [(ac,len(sum_is(dxs,ac))) for ac in range(rl,rh+1) if sum_is(dxs,ac).shape[0]>nz]
    print()
    print(title)
    for ac,ncases in ids:
        print('acs: {}, cases: {}'.format(ac,ncases))
    total = sum([nc for ac,nc in ids])
    print('total: {}'.format(total))
<<<<<<< HEAD
=======
def write_ac_cases(dxs,title,fname):
    ids = [(ac,len(sum_is(dxs,ac))) for ac in range(0,dxs.shape[1]) if sum_is(dxs,ac).shape[0]>0]
    fname = fname if '.txt' in fname else '{}.txt'.format(fname)
    with open (fname,'a') as f:
        f.write('\ntitle')
        for ac,ncases in ids:
            f.write('acs: {}, cases: {}'.format(ac,ncases))
        total = sum([nc for ac,nc in ids])
        f.write('total: {}'.format(total))
    
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12

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
<<<<<<< HEAD
=======
def match_ids_sequence(ids_seq):
    mt_ids = ids_seq[0]
    for sqi_ids in ids_seq[1:]:
        mt_ids = mt_ids[sqi_ids]
    return mt_ids
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12

# same as below, but more exhaustive, with new algorithms
def mk_dxs_symsets(dxs,sxc,ids=False,print_data=True):
    if len(dxs.shape)==2:
        dxs = mk_dxs_tensor(dxs,sxc)
    sxc = expand_domain(sxc) if dxs[0].flatten().shape[0] > sxc.flatten().shape[0] else sxc
    dxs = sort_by_sum(dxs)
    # symset_cases: symset canon id, dx id 
    symset_cases = np.ones((dxs.shape[0],2)).astype(int)*-1
    print()
    for di in tqdm(range(dxs.shape[0])):
        # if already found as an instance of other sx
        if di in symset_cases[:,1]:
            pass
        # take as sx canon
        elif di not in symset_cases[:,0]:
            symset_cases[di] = [di,di]
            # reduce to min size and make variants
            sx = dxs[di]
            if rm_zero_layers(sx).shape != sx.shape:
                sx = expand_domain(rm_zero_layers(sx))
            non_env = False if np.sum(abs((mk_moore_nb(sx)+sx-1))) > 5 else True
            sx_vxs = mk_sx_variants(sx,mk_non_env=non_env)
            # look for repeated instances
            for dj in range(dxs[di+1:].shape[0]):
                dij = di+1+dj
                dxj = expand_domain(dxs[dij])
                # if not already checked before
                if dij in symset_cases:
                    pass
                elif is_sx_in_dx(sx_vxs,dxj):
                    symset_cases[dij] = [di,dij]
    sms_ids = np.array(sorted(list(set(symset_cases[:,0]))))
    sms_dxs = center_tensor_sxs(sort_by_sum(dxs[sms_ids]))
    if print_data:
        print_ac_cases(sms_dxs,title='symsets:')
    if ids:
        symset_cases = np.array(sorted(list(symset_cases),key=lambda x:(x[0],x[1])))
        return sms_dxs,symset_cases
    return sms_dxs
# look for symmetries from less to more activ cells 
# sxs: matrix of gol sts in array form
# sx: sample/canon for reshaping
def mk_symsets(sxs,sx,incremental=False,print_data=True,return_data=False):
    # for expanded cases
    sx = expand_domain(sx) if sxs.shape[1] != sx.flatten().shape[0] else sx
    n,m = sx.shape
    ncells = n*m
    # symsets cases: for each sx array: ac,symset canon/type id, id (easier later)
    sxs = sort_by_sum(sxs)
    symset_cases = np.zeros((sxs.shape[0],3)).astype(int)
    # list of (ac,ids), if 1 make directly, omit all cases for ac=0
    for ac,id1 in [(ac,sum_is(sxs,ac)[0]) for ac in range(ncells) if sum_is(sxs,ac).shape[0]==1]:
        symset_cases[id1] = [ac,id1,id1]
    ac_sxs = [(ac,sum_is(sxs,ac)) for ac in range(ncells) if sum_is(sxs,ac).shape[0]>1]
    # compare sxs sts
    for ac,ac_ids in ac_sxs:
        for ei,idx in enumerate(ac_ids):
            if symset_cases[idx][0] == 0:
                symset_cases[idx] = [ac,idx,idx]
                for idx2 in ac_ids[ei+1:]:
                    if symset_cases[idx2][0] == 0:
                        if are_symmetrical(sxs[idx].reshape(n,m),sxs[idx2].reshape(n,m)):
                            symset_cases[idx2] = [ac,idx,idx2]
    # ids for all arrays in same symset, and indices for only first case (canon-like)
    symset_cases = np.array(sorted(list(symset_cases),key=lambda x:(x[0],x[1]))).reshape(sxs.shape[0],3)
    symset_ids = np.array(sorted(list(set(symset_cases[:,1]))))
    symset_tensor =  mk_dxs_tensor(sxs[symset_ids],sx)
    if print_data:
        print_ac_cases(sxs[symset_ids],title='general symsets:')
    if incremental:
        if return_data:
            symset_tensor,symset_cases,symset_ids = mk_incremental_symsets(sxs,sx,symset_cases,symset_ids,print_data=print_data,return_data=True)
        else:
            symset_tensor = mk_incremental_symsets(sxs,sx,symset_cases,symset_ids,print_data=print_data,return_data=False)
    if return_data: 
        return symset_tensor,symset_cases,symset_ids
    return symset_tensor
# check for equivalent instances within symsets
# basically, patterns with more acs should be different to make a new symset
# e.g., if sx_ac=4 and sx_ac=6 are the same plus 2, then: (sx,ei) and (sx,ej)
def mk_incremental_symsets(sxs,sx,sms_cases,pbs_ids,print_data=True,return_data=True):
    sx = expand_domain(sx) if sxs.shape[1] != sx.flatten().shape[0] else sx
    n,m = sx.shape
    checked_pbs = []
    for ei,pbi in enumerate(pbs_ids):
        if pbi not in checked_pbs:
            checked_pbs.append(pbi)
            for pbj in pbs_ids[ei+1:]:
                if are_sx_instances(sxs[pbi].reshape(n,m),sxs[pbj].reshape(n,m)):
                    pbij = np.where(sms_cases[:,1]==pbj)[0]
                    checked_pbs.append(pbj)
                    sms_cases[pbij,1] = pbi
    sms_cases = np.array(sorted(list(sms_cases),key=lambda x:(x[0],x[1]))).reshape(sxs.shape[0],3)
    pbs_ids = np.array(sorted(list(set(sms_cases[:,1]))))
    if print_data:
        print_ac_cases(sxs[pbs_ids],title='incremental symsets:')
    sms_tensor = mk_dxs_tensor(sxs[pbs_ids],sx)
    if return_data:
        return sms_tensor,sms_cases,pbs_ids
    return sms_tensor
<<<<<<< HEAD
=======
# final version, much faster
def mk_symsets_large_dxs(dxs,print_data=True):
    dxs = sort_by_sum(dxs)
    dxs = center_tensor_sxs(dxs)
    # canon/type, if instance, number of instances
    symset_cases = np.ones((dxs.shape[0],2)).astype(int)*-1
    reps = np.zeros(dxs.shape[0])
    for di in tqdm(range(dxs.shape[0])):
        if reps[di] == 0:
            dx_ids = is_sx_in_dxs(dxs[di],dxs)
            symset_cases[di] = [di,dx_ids.shape[0]]
            reps[dx_ids] = 1
    sms_ids = np.where(symset_cases[:,1]>0)[0]
    dxs = dxs[sms_ids]
    symset_cases = symset_cases[sms_ids]
    if print_data:
        print_ac_cases(dxs,title='symsets:')
    dxs = sort_by_sum(dxs)
    dxs = center_tensor_sxs(dxs)
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
# map min to sms cases for counts and dists
def mk_minmap_counts(sms_cases,min_cases,minmap):
    counts = min_cases*1
    for i in range(min_cases.shape[0]):
        mi_id,mi_sum = min_cases[i]
        mi_sum += np.sum(sms_cases[minmap[mi_id],1])
        counts[i] = [mi_id,mi_sum]
    return counts
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12

# just to get ids as from the whole proto domain
# dxs_ids = array of ids
# sms_cases = matrix of arrays: [ac,idx,idy]
def match_dxs_sms_ids(dxs_ids,sms_cases):
    cases = np.pad(sms_cases.astype(int),(0,2))*1
    cases[:,3] = dxs_ids[cases[:,1]]
    cases[:,4] = dxs_ids[cases[:,2]]
    return cases

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

<<<<<<< HEAD
def rm_zero_layers(xdx):
    dx = xdx*1
=======
# remove all zero rows/cols (assumes sx is centered in dx)
# if squared: remove considering entire row-col layers 
def rm_zero_layers(xdx,squared=False):
    dx = xdx*1
    if squared and np.sum(xdx)>0:
        while squared==False:
            dx = dx[1:-1,1:-1]
            if sum_borders(dx) > 0:
                return dx
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    vs = np.sum(dx,axis=1).nonzero()[0]
    dx = dx[min(vs):max(vs)+1]
    hs = np.sum(dx,axis=0).nonzero()[0]
    dx = dx[:,min(hs):max(hs)+1]
    return dx

# same but quicker
def mk_minimal_sets(symsets,ids=False,print_data=True):
    symsets = sort_by_sum(symsets)
    minsets_cases = np.ones((symsets.shape[0],2)).astype(int)*-1
    min_cases = []
    print()
    for si in tqdm(range(symsets.shape[0])):
        if si in minsets_cases:
            pass
        else:
            minsets_cases[si] = [si,si]
            sym_sx = symsets[si]
            for sj in range(symsets[si+1:].shape[0]):
                sij = si+1+sj
                sym_sy = symsets[sij]
                if is_in_domain(sym_sx,sym_sy):
                    minsets_cases[sij] = [si,sij]
                    min_cases.append([si,sij])
    min_ids = np.array(sorted(list(set(minsets_cases[:,0]))))
    min_dxs = center_tensor_sxs(sort_by_sum(symsets[min_ids]))
    if print_data:
        print_ac_cases(min_dxs,title='minimal/source cases:')
    if ids:
        min_cases = np.array(sorted(min_cases,key=lambda x:(x[0],x[1])))
        # minsets_cases = np.array(sorted(list(minsets_cases),key=lambda x:(x[0],x[1])))
        return min_dxs,min_cases
    return min_dxs
# set of symset domains not contained in any other
# 'prime' proto structures
# symset: tensor of gol domains
def mk_minset(symset,print_data=True,return_data=False):
    # sort increasingly by ac sum
    symset = np.array(sorted(list(symset),key=lambda x:np.sum(x)))
    min_sxs_ids = []
    rep_sxs_ids = []
    min_rep_cases = []
    # for ei,sx in enumerate(symset):
    for ei in tqdm(range(symset.shape[0])):
        sx = symset[ei]
        if ei not in rep_sxs_ids:
            min_sxs_ids.append(ei)
            for ey,sy in enumerate(symset[ei+1:]):
                ej = ei+1+ey
                if return_data:
                    if is_in_domain(sx,sy):
                        rep_sxs_ids.append(ej)
                        min_rep_cases.append([np.sum(sx),ei,ej])
                else:
                    # in this case it doesn't matter if sx appears in many sy
                    if ej not in rep_sxs_ids:
                        if is_in_domain(sx,sy):
                            rep_sxs_ids.append(ej)
    min_sxs_ids = np.array(min_sxs_ids)
    if print_data:
        print_ac_cases(symset[min_sxs_ids],title='minsets:')
    if return_data:
        return symset[min_sxs_ids],np.array(min_rep_cases),min_sxs_ids
    return symset[min_sxs_ids]

# discard composed patterns
# dxs = tensor of matrix domains (if not, requires sx for reshaping)
# returns tensor if input is tensor, otherwise matrix
def check_adjacency(dxs,sx=[],print_data=True,ids=False):
    dxs_ids = []
    tensor = True
    if len(dxs.shape) == 2:
        dxs = mk_dxs_tensor(dxs,sx)
        tensor = False
<<<<<<< HEAD
    for di,dx in enumerate(dxs):
=======
    # for di,dx in enumerate(dxs):
    for di in tqdm(range(dxs.shape[0])):
        dx = dxs[di]
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
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

<<<<<<< HEAD
=======
# same as below, but faster & more precise:
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

>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
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
<<<<<<< HEAD
=======
    if len(ids_bpxs)==0:
        if ids:
            return dxs,sum_nonzero(dxs)
        return dxs
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
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

<<<<<<< HEAD
# for very large symsets dxs
def incremental_symsets(dxs):
    for xi in range(3,7):
        xi_sms,xi_cases = mk_dxs_symsets(sum_is(dxs,xi,arrays=True),ids=True)
        pdb.set_trace()
    sum_in_range()

=======
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
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

<<<<<<< HEAD
=======
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
def mk_txs_matrix(pxs,txs,dims=4):
    # clean discontinuous patterns
    pxtx = []
    for pbi,fpxs in enumerate(pxs):
        for fpi,fpx in enumerate(fpxs):
            if is_composed(fpx):
                fpx_id,px = mk_binary_index(fpx,dims=dims)
                pxtx.append([fpx_id,(pbi,fpi),px,txs[pbi][fpi][1]])
    return pxtx

>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12

# sxs1,sxs2: arrays for gol sts 
# ss1,ss2: symsets from sxs1,sxs2
def check_matching_symsets(sxs,ssx,sxys,ssy,xdims=[0,0],ydims=[0,0]):
    # if not dims, assume squared domains
    if np.sum(xdims)==0:
        xdims = [np.sqrt(sxs[0].shape[0]).astype(int)]*2
    if np.sum(ydims)==0:
        ydims = [np.sqrt(sxys[0].shape[0]).astype(int)]*2
    nx,mx = xdims
    ny,my = ydims
    # output arrray
    matching_ids = []
    # compare according to active cells
    ncells = max(sxs.shape[1],sxys.shape[1])
    for ac in range(ncells):
        if ac in ssx.keys() and ac in ssy.keys():
            for symset_x in ssx[ac]:
                sx = sxs[symset_x[0]].reshape(nx,mx)
                for symset_y in ssy[ac]:
                    sy = sxys[symset_y[0]].reshape(ny,my)
                    if are_symmetrical(sx,sy):
                        matching_ids.append([symset_x[0],symset_y[0]])
    print('\nmatching_ids:')
    for i,j in matching_ids:
        print(i,j)
    return matching_ids

# distance matrices for intrinsic info
# for every x and y value of a,b,...,n elements: sqrt( (ax-bx)**2 + (ay-by)**2 )
# basically the euclidian distance for every comparison
def make_dms(count):
    # transition matrix for x
    # given x, what are the probs for y
    # every value divided by the sum of the rows (values for x)
<<<<<<< HEAD
    tm_x = count/np.sum(count,axis=1)
=======
    # tm_x = count/np.sum(count,axis=1)
    tm_x = count*1
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    # transition matrix for y
    # given y, the probs of x
    # knowing y, it is each value divided by the vertical sum (values of y)
    # then transposed, so it is in function of y->x instead of x->y
<<<<<<< HEAD
    tm_y = (count/np.sum(count,axis=0)).T
=======
    # tm_y = (count/np.sum(count,axis=0)).T
    tm_y = np.ascontiguousarray(count.T)
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
    # distance matrices
    dim = tm_x.shape[0]
    # fill x
    dmx = np.zeros((dim,dim))
    for ei,i in enumerate(tm_x):
        for ej,j in enumerate(tm_x):
            dmx[ei,ej] = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    # fill y 
    dmy = np.zeros((dim,dim))
<<<<<<< HEAD
    for ei,i in enumerate(tm_y):
        for ej,j in enumerate(tm_y):
            dmy[ei,ej] = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    return dmx,dmy
=======
    pdb.set_trace()
    for ei,i in enumerate(tm_y):
        for ej,j in enumerate(tm_y):
            dmy[ei,ej] = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    return dmx,dmy#.T
>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12

# saving and loading with pickle
def save_as(file,name,ext=''):
    import pickle
    import os
    if not ext:
        fname = name if '.' in name else '{}.{}'.format(name,'unk')
    else:
        fname = '{}.{}'.format(name,ext)
    while os.path.isfile(fname):
        i = 1
        name,ext = fname.split('.')
        try:
            fi = int(name[-1])+1
            fname = '{}{}.{}'.format(fname[:-1],fi,ext)
        except:
            fi = i+1
            fname = '{}{}.{}'.format(fname,i,ext)
    with open(fname,'wb') as f:
        pickle.dump(file,f)
    print('\nsaved as: {}\n'.format(fname))

def load_data(filename='',auto=True,ext=''):
    import pickle
    import os
    if filename:
        try:
            with open(filename,'rb') as fname:
                fdata = pickle.load(fname)
                return fdata
        except:
            print('\n{} not in dir\n'.format(filename))
    fnames = [i for i in os.listdir() if '.{}'.format(ext) in i]
    x = 1
    while True==True:
        print()
        for ei,fi in enumerate(fnames):
            print('{} - {}'.format(ei+1,fi))
            if not auto:
                x = int(input('\nfile: _ '))
            try:
                with open(fnames[x-1],'rb') as fname:
                    fdata = pickle.load(fname)
                    return fdata
            except:
                print('\ninvalid input?\n')
        if auto==True:
            print('\ndidn\'t find anything\n')
            auto = False

<<<<<<< HEAD
=======

>>>>>>> d58e35827ffd0d185ecddcfba990b8ebfe162d12
class GolPx:
    def __init__(self,px_name,px=[]):
        self.name = px_name
        self.px = mk_gol_pattern(px_name) if len(px)==0 else px
        # proto/fwd domains
        self.pt_dxs,self.fwd_dxs = [],[]
        self.pt_dxs_ids,self.fwd_dxs_ids = None,None
        # proto/fwd symsets
        self.pt_sms = []
        self.pt_gen_cases = None
        self.pt_sms_cases = None
        # proto/fwd minset
        self.pt_min = []
        self.pt_min_cases = None
        # proto fxs (fwd are fast, dont need save/load)
        self.proto_fxs()
        self.fwd_fxs()
        
    # main fxs
    def proto_fxs(self):
        self.mk_proto_domains()
        self.mk_proto_symsets()
        self.mk_proto_minset()
        self.get_cause_info()
    def fwd_fxs(self):
        self.mk_fwd_domains()
        self.mk_fwd_symsets()
        self.mk_fwd_minset()
        self.get_effect_info()
    # obj proto domains
    def mk_proto_domains(self):
        self.pt_dxs_all = mk_binary_domains(self.px.shape[0]*self.px.shape[1])
        print('\n{} total past domains: {}'.format(self.name,self.pt_dxs_all.shape[0]))
        fname = 'data_{}_proto_domains.gol'.format(self.name)
        self.pt_dxs,self.pt_dxs_ids_bf_ct,self.pt_dxs_ids = self.try_load(fname,nf=3,print_ac=False)
        if self.loaded:
            self.print_ac_cases(self.pt_dxs_all[self.pt_dxs_ids_bf_ct],title='before CT')
            self.print_ac_cases(self.pt_dxs,title='after CT')
        else:
            self.pt_dxs,self.pt_dxs_ids_bf_ct,self.pt_dxs_ids = get_sxs_from_sy(self.px,self.name,return_data=True)
            self.save_data([self.pt_dxs,self.pt_dxs_ids_bf_ct,self.pt_dxs_ids],fname)
    # obj fwd domains
    def mk_fwd_domains(self):
        print("\nfwd domains:")
        self.sx_dxs = mk_sx_domains(self.name)
        # fname = 'data_{}_fwd_domains.gol'.format(self.name)
        self.fwd_dxs_all,self.fwd_dxs,self.fwd_dxs_ids_bf_ct,self.fwd_dxs_ids = get_sxys_from_sx(self.sx_dxs,self.px,return_data=True)
    # fx to make symsets
    def mk_symsets(self,dxs,px,incremental=True):
        sms_tensor,sms_cases,sms_ids = mk_symsets(dxs,px,incremental=incremental,return_data=True)
        return sms_tensor,sms_cases,sms_ids
    # fx to pass from gen to inc symsets
    def mk_inc_symsets(self,dxs,px,gen_sms,gen_ids):
        sms_tensor,sms_cases,sms_ids = mk_incremental_symsets(dxs,px,gen_sms,gen_ids)
        return sms_tensor,sms_cases,sms_ids
    # fx to mach ids
    def match_ids(self,dxs_ids,cases_ids):
        return match_dxs_sms_ids(dxs_ids,cases_ids)
    # obj proto symsets
    def mk_proto_symsets(self):
        fname='data_{}_proto_symsets.gol'.format(self.name)
        self.pt_sms,self.pt_gen_cases,self.pt_sms_cases = self.try_load(fname,nf=3,print_ac=False)
        if self.loaded:
            gen_ids = np.array(list(set(self.pt_gen_cases[:,3])))
            self.print_ac_cases(self.pt_dxs_all[gen_ids],title='general symsets')
            self.print_ac_cases(self.pt_sms,title='incremental symsets')
        else:
            # general
            gen_sms,gen_cases,gen_ids = self.mk_symsets(self.pt_dxs,self.px,incremental=False)
            self.pt_gen_cases = self.match_ids(self.pt_dxs_ids,gen_cases)
            # incremental
            self.pt_sms,sms_cases,sms_ids = self.mk_inc_symsets(self.pt_dxs,self.px,gen_cases,gen_ids)
            self.pt_sms_cases = self.match_ids(self.pt_dxs_ids,sms_cases)
            self.save_data([self.pt_sms,self.pt_gen_cases,self.pt_sms_cases],fname)
    # obj fwd symsets
    def mk_fwd_symsets(self):
        gen_sms,gen_cases,gen_ids = self.mk_symsets(self.fwd_dxs,self.px,incremental=False)
        self.fwd_gen_cases = self.match_ids(self.fwd_dxs_ids,gen_cases)
        self.fwd_sms,sms_cases,sms_ids = self.mk_inc_symsets(self.fwd_dxs,self.px,gen_cases,gen_ids)
        self.fwd_sms_cases = self.match_ids(self.fwd_dxs_ids,sms_cases)
    # fx to make minset
    def mk_minset(self,sms):
        minset,minset_cases,minset_ids = mk_minset(sms,return_data=True)
        return minset,minset_cases,minset_ids
    # obj proto minset
    def mk_proto_minset(self):
        fname = 'data_{}_proto_minset.gol'.format(self.name)
        self.pt_min,self.pt_min_cases = self.try_load(fname,nf=2)
        if not self.loaded:
            self.pt_min,pt_min_cases,pt_min_ids = self.mk_minset(self.pt_sms)
            self.pt_min_cases = self.match_ids(self.pt_dxs_ids,pt_min_cases)
            self.save_data([self.pt_min,self.pt_min_cases],fname)
    # obj fwd_minset
    def mk_fwd_minset(self):
        fname = 'data_{}_fwd_minset.gol'.format(self.name)
        self.fwd_min,self.fwd_min_cases = self.try_load(fname,nf=2)
        if not self.loaded:
            self.fwd_min,fwd_min_cases,fwd_min_ids = self.mk_minset(self.fwd_sms)
            self.fwd_min_cases = self.match_ids(self.fwd_dxs_ids,fwd_min_cases)
            self.save_data([self.fwd_min,self.fwd_min_cases],fname)
    # obj cause info
    def get_cause_info(self):
        pbs_ids = np.array(sorted(list(set(self.pt_min_cases[:,3]))))
        self.crep = pbs_ids*0
        for i,pb_id in enumerate(pbs_ids):
            self.crep[i] = np.where(self.pt_min_cases[:,3]==pb_id)[0].shape[0]
        print('\ncause repertoire (counts):\n{}\ntotal: {}'.format(self.crep,np.sum(self.crep)))
        self.crep = self.crep/np.sum(self.crep)
        uc = np.ones(self.crep.shape)/self.crep.shape[0]
        self.pt_dm = []
        self.ci = 0
    # obj effect info
    def get_effect_info(self):
        fwd_ids = np.array(sorted(list(set(self.fwd_min_cases[:,3]))))
        self.erep = fwd_ids*0
        for i,fwd_id in enumerate(fwd_ids):
            self.erep[i] = np.where(self.fwd_min_cases[:,3]==fwd_id)[0].shape[0]
        print('\neffect repertoire (counts):\n{}\ntotal: {}'.format(self.erep,np.sum(self.erep)))
        self.erep = self.erep/np.sum(self.erep)
        self.ucf = np.ones(self.erep.shape)
        self.pt_dm = []
        self.ei = 0
    # fx for ordering by ac
    def print_ac_cases(self,x,title=''):
        print_ac_cases(x,title=title)
    # fx plot
    def plot(self,x):
        plt.plot(x)
        plt.show()
        plt.close()
    # fx save
    def save_data(self,x,filename):
        name,ext = filename.split('.')
        save_as(x,name,ext)
    # fx load
    def try_load(self,filename,nf=1,print_ac=True,title=''):
        self.loaded = False
        if filename in os.listdir():
            self.loaded = True
            data = load_data(filename)
            if print_ac:
                title = filename if title == '' else title
                self.print_ac_cases(data[0],title=title)
            return data
        return [[]]*nf
        

