
import numpy as np
from gol_auxs import mk_gol_pattern, mk_sx_variants
from gol_fxs import load_data

class GolPattern:
    def __init__(self, name, domains=False):
        self.label = name
        self.sx, self.dx = mk_gol_pattern(self.label, domain=True)
        self.mb = mk_moore_nb(self.dx)                 # membrane
        self.env = mk_moore_nb(self.dx + self.mb)      # environment
        self.mk_rec_vxs()
        if domains:
            self.load_domains()

    def load_domains(self):
        self.dxs, self.dys, self.syms, self.cases, ids = load_data(filename=self.label)
        
    def mk_rec_vxs(self):
        vxs = mk_sx_variants(self.sx, mk_non_env=True)
        self.vxs = np.rec.array(None, dtype=[('sxs',np.ndarray),
                                             ('e0',np.bool_),
                                             ('ecells',np.uint8),
                                             ('id',np.uint64)],
                                             shape = len(vxs))
        for ei,vxi in enumerate(vxs):
            self.vxs[ei].sxs = vxi
            ecells = vxi.nonzero()[0].shape[0] - self.sx.nonzero()[0].shape[0]
            self.vxs[ei].ecells = ecells
            self.vxs[ei].e0 = True if ecells == 0 else False
            self.vxs[ei].id = array2int(vxi)
        
        self.nsxs = len(vxs)

    def sxs0(self):
        print(f'\nenv0 cases: {sum(self.vxs.ecells==False)}\n')
        return self.vxs.sxs[self.vxs.ecells==False]

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

def mk_px_domains(px,cap=10,tensor=True,save=False):
    env_bin_dxs = mk_binary_domains(px.env.sum())
    px_env_dxs = np.zeros((env_bin_dxs.shape[0],px.dx.flatten().shape[0])).astype(int)
    env_ids = px.env.flatten().nonzero()[0]
    for i,env_id in enumerate(env_ids):
        px_env_dxs[:,env_id] = env_bin_dxs[:,i]
    if cap:
        px_env_dxs = px_env_dxs[px_env_dxs.sum(axis=1)<=cap]       
    sx_ids = px.dx.flatten().nonzero()[0]
    px_env_dxs[:,sx_ids] = 1
    if tensor:
        px_env_dxs = mk_tensor(px_env_dxs,px.dx)
    if save:
        fname = f'gol_domains_cap={cap}_{px.label}'
        save_as(px_env_dxs,name=fname)
        return
    return px_env_dxs

# gol transition step (dy: dx expanded for new ON cells)
def gol_tx(dx):
    dx = expand_domain(dx)
    dy = np.zeros(dx.shape).astype(int)
    for ei in range(dx.shape[0]):
        for ej in range(dx.shape[1]):
            nb = dx[max(0,ei-1):ei+2,max(0,ej-1):ej+2].sum() - dx[ei,ej]
            dy[ei,ej] = 1 if nb==3 or dx[ei,ej]*nb==2 else 0
    return dy
def multi_gol_tx(dxs):
    from tqdm import tqdm
    dys = np.zeros((dxs.shape[0],dxs.shape[1]+2,dxs.shape[2]+2)).astype(int)
    for di in tqdm(range(dxs.shape[0])):
        dys[di] = gol_tx(dxs[di])
    return dys


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
    if os.path.isfile(fname):
        print(f'\nfile already exists at {fname}\n')
        return
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
            with open(fnames[int(x)-1],'rb') as fname:
                fdata = pickle.load(fname)
                return fdata
        except:
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

# make domains
# for px in pxs:
#     mk_px_domains(px,save=True)
for px in pxs:
    fname = f'gol_domains_cap=10_{px.label}'
    dxs = load_data(filename=fname)
    dys = multi_gol_tx(dxs)
    fname = f'gol_tx_domains_cap=10_{px.label}'
    save_as(dys,name=fname)

