
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# sx, dx, etc. astype int?
class GolPattern:
    def __init__(self, name):
        self.label = name
        self.sx, self.dx = mk_gol_pattern(self.label, domain=True)
        self.dy = expand_domain(self.dx)
        self.mb = mk_moore_nb(self.dx)                 # membrane
        self.env = mk_moore_nb(self.dx + self.mb)      # environment
        self.id = array2int(self.sx)
        self.mk_rec_variants()
        self.txs = None
        # spiking 
        # self.ons, self.offs, self.ev, self.e = None,None,None,0

    def mk_rec_variants(self):
        vxs = mk_sx_variants(self.sx)
        self.vxs = np.rec.array(None, dtype=[('sx',np.ndarray),
                                             ('mb',np.ndarray),
                                             ('env',np.ndarray),
                                             ('id',np.uint64)],
                                             shape = len(vxs))
        for ei,vxi in enumerate(vxs):
            self.vxs[ei].sx = vxi
            self.vxs[ei].mb = mk_moore_nb(vxi)
            self.vxs[ei].env = mk_moore_nb(expand_domain(vxi+mk_moore_nb(vxi)))
            self.vxs[ei].id = array2int(vxi)



'''basic fxs'''
# make canonical gol patterns (sx,e=0) from word inputs
# minimal form: active cells + moore neighborhood rectangled
def mk_gol_pattern(px,domain=False):
    # active cells = 3
    if px == 'blinker':
        dx = np.zeros((5,3))
        dx[1:-1,1] = 1
        # dx = dx.T
    elif px == 'pb0':
        dx = np.ones((2,2))
        dx[0,1] = 0
        dx = np.pad(dx,(1,1))
    # active cells = 4
    elif px == 'block':
        dx = np.zeros((4,4))
        dx[1:-1,1:-1] = 1
    elif px == 'bar':
        dx = expand_domain(np.ones((1,4)))
    elif px == 'baby':
        dx = expand_domain(np.array([[1,0,1],[0,1,1]]))
    elif px == 'worm':
        dx = expand_domain(np.array([[0,0,1,0],[1,1,0,1]]))
    elif px == 'zigzag':
        dx = expand_domain(np.array([[1,0,1,0],[0,1,0,1]]))
    elif px == 'tetrisL':
        dx = expand_domain(np.ones((1,3)))
        dx[2,3] = 1
        dx = expand_domain(rm_zero_layers(dx))
    elif px == 'tetrisT':
        dx = expand_domain(np.ones((1,3)))
        dx[2,2] = 1
        dx = expand_domain(rm_zero_layers(dx))
    elif px == 'tetrisZ':
        dx = expand_domain(np.array([[0,1,1],[1,1,0]]))
    # also 4, but newer (after alife24)
    elif px == 'tub':
        dx = np.zeros((3,3))
        dx[1,:] = 1
        dx[:,1] = 1
        dx[1,1] = 0
        dx = np.pad(dx,(1,1))
    elif px == 'helix':
        dx = np.zeros((2,4))
        dx[1,:2] = 1
        dx[0,2:] = 1
        dx = np.pad(dx,(1,1))
    elif px == 'boat':
        dx = expand_domain(np.array([[1,0,0,1],[0,1,1,0]]))
    elif px == 'prybar':
        dx = np.zeros((3,3))
        dx[0,1] = 1
        dx[1] = [1,0,1]
        dx[2,2] = 1
        dx = expand_domain(dx)
    # active cells = 5
    elif px == 'gliderA':
        dx = np.zeros((5,5))
        dx[1,2] = 1
        dx[2,3] = 1
        dx[3,1:-1] = 1
    elif px == 'gliderB':
        dx = np.zeros((5,5))
        dx[1,1] = 1
        dx[2:-1,2] = 1
        dx[1:3,3] = 1
    elif px == 'flag':
        dx = expand_domain(np.ones((2,2)))
        dx[3,2] = 1
        dx = expand_domain(rm_zero_layers(dx))
    elif px == 'kite':
        dx = expand_domain(np.ones((2,2)))
        dx[3,3] = 1
        dx = expand_domain(rm_zero_layers(dx))
    # also 5, but later inclusion (after alife24)
    elif px == 'firefly':
        dx = np.zeros((4,4))
        dx[0,3] = 1
        dx[1] = [1,0,1,0]
        dx[2,1] = 1
        dx[3,2] = 1
        dx = expand_domain(dx)
    elif px == 'ufo':
        dx = np.zeros((4,4))
        dx[0,2] = 1
        dx[1] = [0,1,0,1]
        dx[2,0] = 1
        dx[3,1] = 1
        dx = expand_domain(dx)
    else:
        print(f'\n\"{px}\" pattern not defined\n')
    dx = dx.astype(int)
    if domain:
        return dx,expand_domain(dx)
    return dx

# to combine patterns
def append_vxs(label,px,py):
    from copy import deepcopy
    new_px = deepcopy(px)
    new_px.label = label
    n_vxs = px.vxs.shape[0] + py.vxs.shape[0]
    vxs = np.rec.array(None, dtype=[('sx',np.ndarray),
                                    ('mb',np.ndarray),
                                     ('env',np.ndarray),
                                     ('id',np.uint64)],
                                     shape = n_vxs)
    for ei,vxi in enumerate(px.vxs):
        vxs[ei] = vxi
    for ei,vxi in enumerate(py.vxs):
        vxs[ei+px.vxs.shape[0]] = vxi
    new_px.vxs = vxs
    return new_px

# array to int
def array2int(arr):
    xi = np.sum([x<<e for e,x in enumerate(arr.flatten().astype(int))])
    return xi

# fxs that return the number, or the ids of the arrays of a matrix that:
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

# print number of cases for each n active cells in domains
def print_ac_cases(doms,rl=0,rh=0,nonzero=True,title=''):
    nz = 0 if nonzero==True else -1
    if len(doms.shape) == 2:
        max_sum = np.max(doms.sum(axis=(1)))
    else:
        max_sum = np.max(doms.sum(axis=(1,2)))
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

# expand domain size: rows/cols=(bef 0,aft n), else layers
def expand_domain(sx,layers=1,rows=(0,0),cols=(0,0)):
    if np.sum([rows,cols])>0:
        return np.pad(sx,(rows,cols),mode='constant')
    return np.pad(sx,layers,mode='constant')

# to check if some sx is surrended by zeros in some domx 
def sum_borders(domx):
    cx = np.sum(domx[1:-1,1:-1])
    return np.sum(domx)-cx

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

# make moore neighborhood
# sxr: reduced sx (the min rectangle containing all act. cells)
def mk_moore_nb(sxr):
    # sxr = np.pad(sxr,(1,1))
    moore_nb = sxr*1
    for i in range(sxr.shape[0]):
        for j in range(sxr.shape[1]):
            moore_nb[i,j] = np.sum(sxr[max(0,i-1):i+2,max(0,j-1):j+2])
    moore_nb = moore_nb * np.abs(sxr-1)
    return np.where(moore_nb>0,1,0)

# remove surrounding zero layers
def rm_zero_layers(dx,active_only=False):
    vs,hs = dx.nonzero()
    sx = dx[max(0,vs.min()-1):vs.max()+2,max(0,hs.min()-1):hs.max()+2]
    if active_only:
        return sx[1:-1,1:-1]
    return sx

# a tensor for all binary combinations
# n_cells are all the env cells in some domain
def mk_binary_domains(n_cells):
    n_cells = n_cells if type(n_cells)==int else int(n_cells)
    doms = np.zeros((2**n_cells,n_cells)).astype(int)
    for i in tqdm(range(n_cells)):
        f = 2**i
        xi = np.concatenate((np.zeros(f),np.ones(f)))
        n = int(2**n_cells/(2**(i+1)))
        doms[:,-1-i] = np.tile(xi,n)
    return doms

# make all rotation and transposed cases
def mk_sx_variants(sx):
    vxs,vars = [],[]
    # rotations and reflections
    for ri in range(4):
        sxr = np.rot90(sx,ri)
        sxt = np.ascontiguousarray(sxr.T)
        vars.extend([sxr,sxt])
    for var in vars:
        vx_in = False
        for vxi in vxs:
            if np.array_equal(var,vxi):
                vx_in = True
                break
        if not vx_in:
            vxs.append(var.astype(int))
    return vxs

# matrix shaped data; tx: tensor sample for reshaping
def mk_tensor(mx,tx):
    if len(mx.shape) > 2:
        return mx
    return mx.reshape(mx.shape[0],tx.shape[0],tx.shape[1])

# print pxs number of env cell and variants
def print_pxs_ncases(pxs):
    npxs = [[i,px.vxs.size] for i,px in enumerate(pxs)]
    npxs = sorted(npxs, key=lambda x:x[1], reverse=False)
    print()
    for pxi in npxs:
        px = pxs[pxi[0]]
        print(f'{px.label}:{" "*(7-len(px.label))} variants={pxi[1]}, \tsx_cells={px.sx.sum()}, memb={px.mb.sum()}, env={px.env.sum()}')

def backmap_ids(ids,new_ids):
    return ids[new_ids]

def entropy_fx(dist):
    h, hb = 0, 0
    for i in dist:
        if i > 0:
            h += i * np.log(i)
            hb += i * np.log2(i)
    print(f'\nentropy: {-h}, in bits: {-hb}\n')
    return -h, -hb


'''make domains, codomains, transitions'''

def mk_px_domains(px,cap=10,save=False):
    env_bin_dxs = mk_binary_domains(px.env.sum())
    px_env_dxs = np.zeros((env_bin_dxs.shape[0],px.dx.flatten().shape[0])).astype(int)
    env_ids = px.env.flatten().nonzero()[0]
    # for i,env_id in tqdm(enumerate(env_ids)):
    #     px_env_dxs[:,env_id] = env_bin_dxs[:,i]
    for i in tqdm(range(len(env_ids))):
        env_id = env_ids[i]
        px_env_dxs[:,env_id] = env_bin_dxs[:,i]
    if cap:
        px_env_dxs = px_env_dxs[px_env_dxs.sum(axis=1)<=cap]       
    sx_ids = px.dx.flatten().nonzero()[0]
    px_env_dxs[:,sx_ids] = 1
    px_env_dxs = mk_tensor(px_env_dxs,px.dx)
    if save:
        cap = cap if cap else f'no_cap={px.env.sum()}'
        fname = f'gol_domains_cap={cap}_{px.label}'
        save_as(px_env_dxs,name=fname)
        return
    return px_env_dxs

# gol transition step (dy: dx expanded for new ON cells)
def gol_tx(dx,expand_dy=True):
    dx = expand_domain(dx) if expand_dy else dx.copy()
    # if expand_dy:
    #     dx = expand_domain(dx)
    dy = np.zeros(dx.shape).astype(int) 
    for ei in range(dx.shape[0]):
        for ej in range(dx.shape[1]):
            nb = dx[max(0,ei-1):ei+2,max(0,ej-1):ej+2].sum() - dx[ei,ej]
            # cell = 0 or 1 and nb=3, or cell=1 and nb=2
            dy[ei,ej] = 1 if nb==3 or dx[ei,ej]*nb==2 else 0
    return dy
def multi_gol_tx(dxs, expand_dy=True):
    if expand_dy:
        dys = np.zeros((dxs.shape[0],dxs.shape[1]+2,dxs.shape[2]+2)).astype(int)
    else:
        dys = np.zeros((dxs.shape[0],dxs.shape[1],dxs.shape[2])).astype(int)
    for di in tqdm(range(dxs.shape[0])):
        dys[di] = gol_tx(dxs[di],expand_dy=expand_dy)
    return dys

# mk all txs from domain
def mk_px_codomains(px,cap=10,save=True):
    cap = cap if cap else f'no_cap={px.env.sum()}'
    fname = f'gol_domains_cap={cap}_{px.label}'
    dxs = load_data(filename=fname)
    dxys = multi_gol_tx(dxs)
    if save:
        fname = f'gol_tx_domains_cap={cap}_{px.label}'
        save_as(dxys,name=fname)
        return
    return dxys

# search for patterns in domain(s)
def is_sx_in_dx(sx,dx,search_borders=False):
    if sx.flatten().shape[0] > dx.flatten().shape[0]:
        return False
    if search_borders:
        dx = expand_domain(dx)
    for wi in range(dx.shape[0] - sx.shape[0]+1):
        for wj in range(dx.shape[1] - sx.shape[1]+1):
            if np.array_equal(dx[wi:wi+sx.shape[0],wj:wj+sx.shape[1]],sx):
                return True
    return False
def is_sx_in_dxs(sx,dxs,search_borders=False):
    if len(dxs.shape) == 2:
        return is_sx_in_dx(sx,dxs,search_borders=search_borders)
    if search_borders:
        dxs = np.pad(dxs,((0,0),(1,1),(1,1)))
    ids = np.zeros(dxs.shape[0]).astype(int)
    for wi in range(dxs.shape[1] - sx.shape[0]+1):
        for wj in range(dxs.shape[2] - sx.shape[1]+1):
            wids = np.zeros(dxs.shape[0])
            wx = dxs[:,wi:wi+sx.shape[0],wj:wj+sx.shape[1]]
            wids[np.sum(wx*sx,axis=(1,2))==sx.sum()] += 0.5
            wids[np.sum(wx*mk_moore_nb(sx),axis=(1,2))==0] += 0.5
            ids += wids.astype(int)
    return ids.nonzero()[0]
# same, but using object
def is_px_in_dx(px,dx,search_borders=False):
    for sx in px.vxs.sx:
        if is_sx_in_dx(sx,dx,search_borders=search_borders):
            return True
    return False
# search px in all domain, no ct restrictions
# sliding window matching sx (2d) in all dxs (3d)
# optional pad (for pxs ct would be false in padded borders)
def is_px_in_dxs(px,dxs,search_borders=False,vxs_ids=False):
    if search_borders:
        dxs = np.pad(dxs,((0,0),(1,1),(1,1)))
    ids = np.zeros(dxs.shape[0]).astype(int)
    for i in tqdm(range(px.vxs.size)):
        sx = px.vxs[i].sx
        mb = px.vxs[i].mb
        for wi in range(dxs.shape[1] - sx.shape[0]+1):
            for wj in range(dxs.shape[2] - sx.shape[1]+1):
                wids = np.zeros(dxs.shape[0])
                wx = dxs[:, wi:wi+sx.shape[0], wj:wj+sx.shape[1]]
                # sx is there, memb unknown; memb=0
                wids[np.sum(wx*sx,axis=(1,2))==sx.sum()] += 0.5
                wids[np.sum(wx*mb,axis=(1,2))==0] += 0.5
                ids += wids.astype(int)
        if vxs_ids:
            ids *= px.vxs[i].id
            return ids
    return ids
    
# dx/dy ids for px -> py transition
# search (py) patterns in codomains (dxys) (i.e., find transitions)
def find_pxpy_txs(px,pxs, cap=False):
    # px pattern codomain
    from copy import deepcopy
    pxs_cp = deepcopy(pxs)
    cap = cap if cap else f'no_cap={px.env.sum()}'
    codomain_fname = f'gol_tx_domains_cap={cap}_{px.label}'
    dxys = load_data(filename=codomain_fname)
    di,dj = dxys.shape[1:]
    px_txs = {}
    # look for patterns
    for py in pxs_cp:
        print(f'\n{px.label} -> {py.label:}')
        ids = np.zeros(dxys.shape[0]).astype(int)
        # for each variant of the pattern
        for i in tqdm(range(py.vxs.size)):
            sy = py.vxs[i].sx
            mb = py.vxs[i].mb
            sy_i,sy_j = sy.shape
            # reshape to omit sy structures non overlapping original sx
            vi,vj = (np.array(dxys.shape[1:] - np.array(sy.shape) -2)/2).astype(int)
            dxy = dxys[:,vi:di-vi,vj:dj-vj]
            dxy_i,dxy_j = dxy.shape[1:]
            # sliding tensor window
            for wi in range(dxy_i-sy_i+1):
                for wj in range(dxy_j-sy_j+1):
                    wx = dxys[:,wi:wi+sy.shape[0],wj:wj+sy.shape[1]]
                    # sx is there, memb is there (sum memb cells = 0)
                    wids = np.zeros(dxy.shape[0])
                    wids[np.sum(wx*sy,axis=(1,2))==sy.sum()] += 0.5
                    wids[np.sum(wx*mb,axis=(1,2))==0] += 0.5
                    ids += wids.astype(int)
        # is there more than 1 sx in dx?
        # if ids[ids>1].shape[0] > 0:
        #     print(f'\nmore than one sx?:\n{ids[ids>1].shape[0]} cases\n{ids[ids>1]}')
        #     examples = np.where(ids>1)[0][:5]
        #     for exi in examples:
        #         print(f'\n{exi}')
        #         print(dxys[exi])
        px_txs[py.label] = np.where(ids>0)[0]
    # disintegrations
    px_txs['end'] = np.where(np.sum(dxys[:,2:-2,2:-2]*px.sx,axis=(1,2))<1)[0]
    # info
    all_txs = dxys.shape[0]
    ntxs = 0
    print()
    for key in px_txs.keys():
        ntxs += px_txs[key].shape[0]
        print(f'{key}: {px_txs[key].shape[0]}')
    print(f'{ntxs} txs known, {all_txs - ntxs}/{all_txs} unknown')
    fname = f'gol_px_txs_cap={cap}_{px.label}'
    save_as(px_txs,name=fname)

def mk_px_data(pxs):
    for px in pxs:
        print(f'\n{px.label}')
        is_dx, is_dxy, is_txs = check_px_data(px, cap=False)
        print(f'dxs data: {is_dx}, co-dxs: {is_dxy}, txs data: {is_txs}')
        if not is_dx:
            print(f'mk {px.label} bin and env domains:')
            mk_px_domains(px, cap=False, save=True)
        if not is_dxy:
            mk_px_codomains(px, cap=False, save=True)
        if not is_txs:
            find_pxpy_txs(px,pxs, cap=False)

# transition map (no caps)
def mk_tx_map(pxs):
    # container dict
    txmap = {}
    for px in pxs:
        txmap[px.label] = {}
        for py in pxs:
            txmap[px.label][py.label] = {}
    # load data
    for px in pxs:
        cap = f'no_cap={px.env.sum()}'
        print(f'\npx: {px.label}, cap = {px.env.sum()}')
        px_dxs = load_data(filename=f'gol_domains_cap={cap}_{px.label}')
        px_dxys = load_data(filename=f'gol_tx_domains_cap={cap}_{px.label}')
        px_txs = load_data(filename=f'gol_px_txs_cap={cap}_{px.label}')
        for py_label in px_txs.keys():
            if py_label == 'end':
                pass
            else:
                # n txs, ids, dx domain, dy domain, env sets/categories
                txmap[px.label][py_label]['nt'] = px_txs[py_label].shape[0]
                txmap[px.label][py_label]['ids'] = px_txs[py_label]
                txmap[px.label][py_label]['dx'] = px_dxs[px_txs[py_label]]
                txmap[px.label][py_label]['dy'] = px_dxys[px_txs[py_label]]
                txmap[px.label][py_label]['ek'] = px_dxs[px_txs[py_label]].sum(axis=(0)) * px.env
    save_as(txmap,name='gol_txmap_no_cap')

'''plotting, sorting data, etc'''

def get_tx_counts(txmap,print_data=True,as_df=False,to_csv=False):
    import pandas as pd
    txs = {}
    for px in txmap.keys():
        txs[px] = {}
        if print_data:
            print()
        px_txs = 0
        for py in txmap[px].keys():
            nt = txmap[px][py]['nt']
            txs[px][py] = nt
            if print_data:
                print(f'{px} -> {py} txs = {nt}')
            px_txs += nt
        if print_data:
            print(f'{px} total txs = {px_txs}')
    if to_csv==True or table==True:
        # pandas sorts it backwards (transposed)
        pd_txs = pd.DataFrame.from_dict(txs).transpose()
        if to_csv:
            pd_txs.to_csv('txs.csv')
    if as_df:
        return pd_txs
    return txs

palette = np.array([[255, 255, 255],   # 0:white
                    [  0, 255,   0],   # 1:green
                    [  0,   0, 255],   # 2:blue
                    [255,   0,   0],   # 3:red
                    [255, 255,   0],   # 4:yellow
                    [255, 128,   0],   # 5:orange
                    [255, 153, 255],   # 6:pink
                    [160,  32, 240],   # 7:purple
                    [128, 128, 128],   # 8:gray
                    [  0,   0,   0]])  # 9:black

def plot_px(px, title='', colors=(1, 9, 2, 0)):
    fig,ax = plt.subplots(1,1)
    sys_color, memb_color, env_color, out_color = colors
    dx = px.dx*sys_color + px.mb*memb_color + px.env*env_color
    imx = ax.imshow(palette[dx.astype(int)])
    title = title if title else f'{px.label} domain'
    # ax.grid(visible=True)
    ax.set_title(title, color='black')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return dx

def plot_pxs(pxs,rows=0,cols=0,horizontal=False,
                                size=(0,0),
                                title='', 
                                colors=(1, 9, 2, 0)):
    if horizontal:
        rows = 1 if rows == 0 else rows
        cols = len(pxs) if cols == 0 else cols
    elif rows+cols == 0:
        cols = int(len(pxs)/2)
        rows = 2
    if sum(size) > 0:
        fig,axs = plt.subplots(nrows=rows,ncols=cols,
                                        figsize=size)
    else:
        fig,axs = plt.subplots(nrows=rows,ncols=cols)
    sys_color, memb_color, env_color, out_color = colors
    # plt.axis('off')
    for ax,px in zip(axs.flat,pxs):
        if px.dx.shape[0] > px.dx.shape[1]:
            dx = px.dx.T*sys_color + px.mb.T*memb_color + px.env.T*env_color
        else:
            dx = px.dx*sys_color + px.mb*memb_color + px.env*env_color
        ax.imshow(palette[dx.astype(int)])#,cmap='binary')
        # ax.grid(visible=True)
        ax.set_title(px.label)
        ax.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout
    plt.show()

def reconstruct_nonzeros(nzs,grid_size=25):
    dxs = np.zeros((len(nzs),grid_size,grid_size))
    for i,nzi in enumerate(nzs):
        dxs[i][nzi] = 1
    return dxs

def gol_animplot(iims, rows=0, cols=0, step=100, color='gray'):
    from matplotlib import animation
    if not isinstance(iims, list):
        iims = [iims]
    nf = iims[0].shape[0]
    # if only one (make spikes)
    if len(iims) == 1:
        cp = iims[0].copy()
        cp[1:] = iims[0][1:] - iims[0][:-1]
        iims.append(cp)
    if rows + cols == 0:
        rows = 1 if len(iims) <= 2 else 2
        cols = 2 if len(iims) <= 2 else int(len(iims)/2) + len(iims)%2
    fig, axs = plt.subplots(rows,cols, figsize=(10,5))
    ti = 0
    ims = []
    fig.suptitle('')
    for ei,ax in enumerate(axs.flat):
        # im = ax.imshow(iims[ei][ti], cmap=color, aspect='auto', animated=True)
        im = ax.imshow(palette[iims[ei][ti].astype(int)])
        ims.append(im)
    def update_fig(ti):
        ti = (ti+1)%nf
        fig.suptitle(f'{ti+1}/{nf}')
        for ui,ax in enumerate(axs.flat):
            iims[ui][ti] = np.where(iims[ui][ti]==-1,3,iims[ui][ti])
            # ax.imshow(palette[iims.astype(int)])
            # import pdb; pdb.set_trace()
            ims[ui].set_array(palette[iims[ui][ti].astype(int)])
            # ims[ui].set_array(iims[ui][ti])
        return [im for im in ims]
    ani = animation.FuncAnimation(fig,update_fig,interval=step,blit=False,repeat=True,cache_frame_data=False)
    plt.show()
    plt.close()



'''save, load, etc'''
def save_as(file,name, fdir='gol_exps_data',ext='gol_info'):
    import pickle
    import os
    if not ext:
        fname = name if '.' in name else '{}.{}'.format(name,'unk')
    else:
        fname = '{}.{}'.format(name,ext)
    fdir = os.path.abspath(os.path.join(os.getcwd(),'..','gol_exps_data'))
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    fname = os.path.join(fdir,fname)
    while os.path.isfile(fname):
        print(f'\nfile already exists at {fname}\n')
        fname += '_(new)'
        # return
    with open(fname,'wb') as f:
        pickle.dump(file,f)
    print('\nsaved as: {}\n'.format(fname))

def load_data(filename='', ext='gol_info',dirname='gol_exps_data'):
    import pickle
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..',dirname)) if dirname else os.getcwd()
    if filename:
        if ext:
            filename += f'.{ext}'
        fpath = os.path.join(dirpath,filename)
        try:
            with open(fpath,'rb') as fname:
                fdata = pickle.load(fname)
                return fdata
        except:
            print('\n{} not as path {}\n'.format(filename,fpath))
    fnames = [i for i in os.listdir(dirpath) if '.{}'.format(ext) in i]
    while True:
        print()
        for ei,fi in enumerate(fnames):
            print('{} - {}'.format(ei+1,fi))
        print()
        x = input('\nfile: _ ')
        if x == 'q' or x == 'quit':
            return
        try:
            fpath = os.path.join(dirpath,fnames[int(x)-1])
            with open(fpath,'rb') as fname:
                fdata = pickle.load(fname)
                return fdata
        except:
            fpath = os.path.join(dirpath,fnames[int(x)-1])
            import pdb; pdb.set_trace()
            print('\ninvalid input? (q/quit to quit)\n')

# check, domain, codomain, txs data
# assuming constrained search (ct_) for txs
def check_px_data(px, cap=False, ext='gol_info',dirname='gol_exps_data'):
    cap = cap if cap else f'no_cap={px.env.sum()}'
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..',dirname)) if dirname else os.getcwd()
    domain_fname = f'gol_domains_cap={cap}_{px.label}.{ext}'
    codomain_fname = f'gol_tx_domains_cap={cap}_{px.label}.{ext}'
    txs_fname = f'gol_px_txs_cap={cap}_{px.label}.{ext}'
    are_files = [os.path.isfile(os.path.join(dirpath,fname)) for fname in [domain_fname,codomain_fname,txs_fname]]
    return are_files

# just shorter, assuming filepath and no_cap files
def load_dxs(px):
    return load_data(filename=f'gol_domains_cap=no_cap={px.env.sum()}_{px.label}')
def load_codxs(px):
    return load_data(filename=f'gol_tx_domains_cap=no_cap={px.env.sum()}_{px.label}')
def load_txs(px):
    return load_data(filename=f'gol_px_txs_cap=no_cap={px.env.sum()}_{px.label}')

    

    

'''run some stuff'''

blinker = GolPattern('blinker')
pb0 = GolPattern('pb0')
block = GolPattern('block')
gla = GolPattern('gliderA')
glb = GolPattern('gliderB')
# for alife24
ttz = GolPattern('tetrisZ')
ttt = GolPattern('tetrisT')
ttl = GolPattern('tetrisL')
zz = GolPattern('zigzag')
bar = GolPattern('bar')
baby = GolPattern('baby')
flag = GolPattern('flag')
kite = GolPattern('kite')
worm = GolPattern('worm')
# new, after alife24
tub = GolPattern('tub')
helix = GolPattern('helix')
boat = GolPattern('boat')
# pbar = GolPattern('prybar')                       # 128 variants and env=24
# these have 2048 variants each
# ffly = GolPattern('firefly')
# ufo = GolPattern('ufo')
glider = append_vxs('glider',gla,glb)

# pxs = [blinker,pb0,block,gla,glb,                   # basic
#        ttz,ttt,ttl,zz,bar,baby,flag,kite,worm,      # alife24
#        tub,helix,boat]                              # wivace
