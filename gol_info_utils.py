
import numpy as np
import matplotlib.pyplot as plt
from gol_auxs import mk_gol_pattern, mk_sx_variants
from tqdm import tqdm

# sx, dx, etc. astype int?
class GolPattern:
    def __init__(self, name, txs=False):
        self.label = name
        self.sx, self.dx = mk_gol_pattern(self.label, domain=True)
        self.mb = mk_moore_nb(self.dx)                 # membrane
        self.env = mk_moore_nb(self.dx + self.mb)      # environment
        self.mk_rec_vxs()
        # self.txs = {}
        self.all_txs, self.ntxs = 0, 0
        self.env_sets = {}
        if txs:
            self.get_txs_data()

    def mk_rec_vxs(self):
        vxs = mk_sx_variants(self.sx, mk_non_env=True)
        self.vxs = np.rec.array(None, dtype=[('sx',np.ndarray),
                                             ('e0',np.bool_),
                                             ('ecells',np.uint8),
                                             ('id',np.uint64)],
                                             shape = len(vxs))
        for ei,vxi in enumerate(vxs):
            self.vxs[ei].sx = vxi
            ecells = vxi.nonzero()[0].shape[0] - self.sx.nonzero()[0].shape[0]
            self.vxs[ei].ecells = ecells
            self.vxs[ei].e0 = True if ecells == 0 else False
            env_mask = np.ones((vxi.shape))
            env_mask[1:-1,1:-1] = 0     # so sx & sx+env have the same id
            self.vxs[ei].id = array2int(vxi*env_mask)
        
        self.nvxs = len(vxs)
        self.sxs0 = self.vxs.sx[self.vxs.ecells==False]

    def get_txs_data(self):
        fname = f'gol_px_txs_ct_{self.label}'
        self.txs = load_data(filename=fname)
        # clean txs info: CT in dy?
        self.txs_info()
        self.mk_env_sets()
        
    def txs_info(self):
        print('{}'.format(f'\n{self.label} ->' if len(self.txs.keys())>0 else '\nno txs info\n'))
        self.ntxs = 0
        for key in self.txs.keys():
            self.ntxs += self.txs[key].shape[0]
            print(f'{key}: {self.txs[key].shape[0]}')
        print(f'{self.ntxs} txs known')
        if self.all_txs:
            print(f'{self.all_txs - self.ntxs}/{self.all_txs} txs unknown')

    def mk_env_sets(self):
        dxs_fname = f'gol_domains_cap=10_{self.label}'
        # dys_fname = f'gol_tx_domains_cap=10_{px.label}'
        dxs = load_data(filename=dxs_fname)
        # dys = load_data(filename=dys_fname)
        self.all_txs = dxs.shape[0]
        for key in self.txs.keys():
            dxs_sums = np.zeros((self.dx.shape)).astype(int)
            for dx_id in self.txs[key]:
                dxs_sums += dxs[dx_id]
            dxs_sums *= np.abs(self.dx - 1).astype(int)
            self.env_sets[key] = dxs_sums/dxs_sums.sum()

    def plot_domains(self):
        mk_plots = True
        while mk_plots:
            print(f'\n{self.label} ->')
            for ei,key in enumerate(self.env_sets.keys()):
                print(f'[{ei}] -> {key}, ncases = {self.txs[key].shape[0]}')
            kx = input('\n? _ ')
            if kx == 'q' or kx == 'quit':
                mk_plots = False
            else:
                try:
                    # info
                    # plots
                    kx_label = list(self.env_sets.keys())[int(kx)]
                    domain = self.env_sets[kx_label]
                    # cmap = mpl.cm.get_cmap("jet").copy()
                    non_dx = self.dx + self.mb + self.env - 1
                    domain += non_dx
                    domain += self.mb * -1
                    domain += self.dx * 2
                    ex_dx = self.env * domain
                    imx = plt.imshow(domain, vmin=0, vmax=np.max(ex_dx), cmap='magma', aspect='auto')
                    imx.cmap.set_over('black')
                    imx.cmap.set_under('white')
                    plt.colorbar()
                    plt.title(f'{self.label} -> {kx_label} environmental category')
                    plt.show()
                except:
                    import pdb; pdb.set_trace()

# array to int
def array2int(arr):
    xi = np.sum([x<<e for e,x in enumerate(arr.flatten().astype(int))])
    return xi

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

# a tensor for all binary combinations
# n_cells are all the env cells in some domain
def mk_binary_domains(n_cells):
    n_cells = n_cells if type(n_cells)==int else int(n_cells)
    doms = np.zeros((2**n_cells,n_cells)).astype(int)
    for i in range(n_cells):
        f = 2**i
        xi = np.concatenate((np.zeros(f),np.ones(f)))
        n = int(2**n_cells/(2**(i+1)))
        doms[:,-1-i] = np.tile(xi,n)
    return doms

# matrix shaped data; tx: tensor sample for reshaping
def mk_tensor(mx,tx):
    if len(mx.shape) > 2:
        return mx
    return mx.reshape(mx.shape[0],tx.shape[0],tx.shape[1])

def mk_px_domains(px,cap=10,save=False):
    env_bin_dxs = mk_binary_domains(px.env.sum())
    px_env_dxs = np.zeros((env_bin_dxs.shape[0],px.dx.flatten().shape[0])).astype(int)
    env_ids = px.env.flatten().nonzero()[0]
    for i,env_id in enumerate(env_ids):
        px_env_dxs[:,env_id] = env_bin_dxs[:,i]
    if cap:
        px_env_dxs = px_env_dxs[px_env_dxs.sum(axis=1)<=cap]       
    sx_ids = px.dx.flatten().nonzero()[0]
    px_env_dxs[:,sx_ids] = 1
    px_env_dxs = mk_tensor(px_env_dxs,px.dx)
    if save:
        fname = f'gol_domains_cap={cap}_{px.label}'
        save_as(px_env_dxs,name=fname)
        return
    return px_env_dxs

# def delta_xy(dx,dy):
#     dxy = np.abs(dy - dx)
#     return dxy

# gol transition step (dy: dx expanded for new ON cells)
def gol_tx(dx):
    dx = expand_domain(dx)
    dy = np.zeros(dx.shape).astype(int)
    for ei in range(dx.shape[0]):
        for ej in range(dx.shape[1]):
            nb = dx[max(0,ei-1):ei+2,max(0,ej-1):ej+2].sum() - dx[ei,ej]
            # cell = 0 or 1 and nb=3, or cell=1 and nb=2
            dy[ei,ej] = 1 if nb==3 or dx[ei,ej]*nb==2 else 0
    return dy
def multi_gol_tx(dxs):
    dys = np.zeros((dxs.shape[0],dxs.shape[1]+2,dxs.shape[2]+2)).astype(int)
    for di in tqdm(range(dxs.shape[0])):
        dys[di] = gol_tx(dxs[di])
    return dys

# sliding window matching sx (2d) in all dxs (3d)
# optional pad (for pxs ct would be false in padded borders)
def is_sx_in_dxs(sx,dxs,pad=False):
    if pad:
        dxs = np.pad(dxs,((0,0),(1,1),(1,1)))
    sxsum = sx.sum()
    # sx_nz = np.ones((sx.shape)).astype(int) - sx
    ids = np.zeros(dxs.shape[0]).astype(int)
    for wi in range(dxs.shape[1]-sx.shape[0]+1):
        for wj in range(dxs.shape[2]-sx.shape[1]+1):
            wids = np.zeros(dxs.shape[0])
            wx = dxs[:,wi:wi+sx.shape[0],wj:wj+sx.shape[1]]
            # sx is there, domain unknown; only sx in domain
            wids[np.sum(wx*sx,axis=(1,2))==sxsum] += 0.5
            wids[wx.sum(axis=(1,2))==sxsum] += 0.5
            # remove if env is non zero 
            # wids[np.sum(wx*sx_nz,axis=(1,2))>0] = 0
            ids += wids.astype(int)            
    return ids

# dx/dy ids for px -> py transition
def mk_pxpy_txs(px,pxs, dx_contraint=True):
    px_fname = f'gol_tx_domains_cap=10_{px.label}'
    px_dys = load_data(filename=px_fname)
    # if dx_contraint:
    di,dj = px_dys.shape[1:]
    vi,vj = 0,0
        # patterns outside dx space are made from env
        # px_dys = px_dys[:,1:-1,1:-1]
    px_txs = {}
    for py in pxs:
        print(f'\n{px.label} -> {py.label:}')
        ids = np.zeros(px_dys.shape[0]).astype(int)
        for i in tqdm(range(py.vxs.sx.shape[0])):
            sy = py.vxs[i].sx
            # patterns outside dx space are made from env
            if dx_contraint:
                vi,vj = (np.array(px_dys.shape[1:] - np.array(sy.shape) -2)/2).astype(int)
            ids += is_sx_in_dxs(sy,px_dys[:,vi:di-vi,vj:dj-vj])
            # ids += is_sx_in_dxs(sy,px_dys)
        # is there more than 1 sx in dx?
        if ids[ids>1].shape[0] > 0:
            print(f'\nmore than one sx?:\n{ids[ids>1].shape[0]} cases\n{ids[ids>1]}')
            examples = np.where(ids>1)[0][:5]
            for exi in examples:
                print(f'\n{exi}')
                print(px_dys[exi])
            # import pdb; pdb.set_trace()
        px_txs[py.label] = np.where(ids>0)[0]
    # disintegrations
    px_txs['end'] = np.where(np.sum(px_dys[:,2:-2,2:-2]*px.sx,axis=(1,2))<1)[0]
    # info
    all_txs = px_dys.shape[0]
    ntxs = 0
    print()
    for key in px_txs.keys():
        ntxs += px_txs[key].shape[0]
        print(f'{key}: {px_txs[key].shape[0]}')
    print(f'{ntxs} txs known, {all_txs - ntxs}/{all_txs} unknown')
    # return np.where(ids>0)[0]
    # save_as(px_txs,name=f'gol_px_txs_{px.label}')
    fname = 'gol_px_txs_{}{}'.format('ct_' if dx_contraint else '', px.label)
    save_as(px_txs,name=fname)

def mk_omap(pxs):
    omap = {}
    labels = []
    for px in pxs:
        labels.append(px.label)
        omap[px.label] = {}
        omap[px.label]['fwd'] = {}
        omap[px.label]['back'] = {}
        omap[px.label]['nfwd'] = {}
        omap[px.label]['nback'] = {}
        omap[px.label]['edx'] = {}
    print()
    for px in pxs:
        for li in tqdm(range(len(labels))):
            py_label = labels[li]
            omap[px.label]['fwd'][py_label] = px.txs[py_label]
            omap[py_label]['back'][px.label] = px.txs[py_label]
            omap[px.label]['nfwd'][py_label] = px.txs[py_label].shape[0]
            omap[py_label]['nback'][px.label] = px.txs[py_label].shape[0]
            omap[px.label]['edx'][py_label] = px.env_sets[py_label]
    return omap

# sx -> sy : tx ids : env set
# sx -> sy : n txs : env set weight/cardinality
# sx <- sy : tx ids : backwards env set?
# sx <- sy : n txs : backwards weight/cardinality
# sx -> sy : dom x
# sx -> sy : dom y
# sx -> sy : delta sx,sy
# sx -> sy : eks : environmental distinctions
# sx -> sy : dom y sums
def mk_delta_map(pxs,omap):
    # make dicts
    dmap = {}
    dmap['labels'] = list(omap.keys())
    for px in pxs:
        dmap[px.label] = {}
        for ku in ['fwd', 'nfwd', 'back', 'nback', 'env_dks', 'dxs', 'dys', 'sxy', 'dysums']:
            dmap[px.label][ku] = {}
    # direct copy from omap
    for px in pxs:
        for py_label in dmap['labels']:
            # fwd tx ids, n, bwd tx ids, n, env distinctions
            for txu in ['fwd', 'nfwd', 'back', 'nback']:
                dmap[px.label][txu][py_label] = omap[px.label][txu][py_label]
                dmap[px.label]['env_dks'][py_label] = omap[px.label]['edx'][py_label]
    # load domains data
    for px in pxs:
        px_dxs = load_data(filename=f'gol_domains_cap=10_{px.label}')
        px_dys = load_data(filename=f'gol_tx_domains_cap=10_{px.label}')
        px_dys_ct = px_dys[:,1:-1,1:-1]
        pxdx = px.dx.astype(int)
        for py_label in dmap['labels']:
            # sx domains, sy domains, sxy transition, sy domain sums 
            dxs_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0],pxdx.shape[1])).astype(int)
            dys_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0]+2,pxdx.shape[1]+2)).astype(int)
            sxy_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0],pxdx.shape[1])).astype(int)
            for ei,id in enumerate(omap[px.label]['fwd'][py_label]):
                dxs_tensor[ei] = px_dxs[id]
                dys_tensor[ei] = px_dys[id]
                sxy_tensor[ei] = pxdx + px_dys_ct[id]
            dmap[px.label]['dxs'][py_label] = dxs_tensor
            dmap[px.label]['dys'][py_label] = dys_tensor
            dmap[px.label]['sxy'][py_label] = sxy_tensor
            dmap[px.label]['dysums'] = dys_tensor.sum(axis=0)[1:-1,1:-1]
    fname = f'gol_delta_map_cap=10'
    save_as(dmap, name=fname)
    # return dmap

'''save, load, etc'''
def save_as(file,name,fdir='gol_exps_data',ext='gol_info'):
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

def load_data(filename='',ext='gol_info',dirname='gol_exps_data'):
    import pickle
    import os
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


blinker = GolPattern('blinker')
pb0 = GolPattern('pb0')
block = GolPattern('block')
gla = GolPattern('gliderA')
glb = GolPattern('gliderB')
ttz = GolPattern('tetrisZ')
ttt = GolPattern('tetrisT')
ttl = GolPattern('tetrisL')
zz = GolPattern('zigzag')
bar = GolPattern('bar')
baby = GolPattern('baby')
flag = GolPattern('flag')
kyte = GolPattern('kyte')
worm = GolPattern('worm')

pxs = [blinker,pb0,block,gla,glb,ttz,ttt,ttl,zz,bar,baby,flag,kyte,worm]

# 1) make domains (dxs)
# for px in pxs:
#     mk_px_domains(px,save=True)

# 2) make co-domains (dys) (GoL txs)
# for px in pxs:
#     fname = f'gol_domains_cap=10_{px.label}'
#     dxs = load_data(filename=fname)
#     dys = multi_gol_tx(dxs)
#     fname = f'gol_tx_domains_cap=10_{px.label}'
#     save_as(dys,name=fname)

# 3) get txs (dxs -> dys) data (labels, ids & n txs)
# from copy import deepcopy
# pxs_cp = [deepcopy(px) for px in pxs]
# for px in pxs:
#     mk_pxpy_txs(px,pxs_cp)

# 4) make txs map, dict for graph-like data
# (change txs=True in GolPattern class)
# omap = mk_omap(pxs)
# fname = f'gol_omap_cap=10'
# save_as(omap, name=fname)

# dxs = load_data(filename=f'gol_tx_domains_cap=10_{px.label}')
# omap = load_data(filename='gol_omap_cap=10')

# 5) make delta(sx,sy) for info comparisons & put all together
# mk_delta_map(pxs,omap)
dmap = load_data(filename='gol_delta_map_cap=10')