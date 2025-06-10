
from gol_info_utils import *
import numpy as np
from collections import defaultdict
# victor purpura stuff
from quantities import s
from neo.core import SpikeTrain
from elephant.spike_train_dissimilarity import victor_purpura_distance as vpdist


'''
see feasibility
pipeline:
1) make tensors and convert to spikes
2) complexity score value
3) victor purpura energy measures
    # define q
4) environmental pressure
    a) txs 10 times-steps for every px/env 
'''

# px: pattern (blinker,block,etc), grid (states) & grid txs tensors
# 10 timesteps, so 11 grid states and 10 grid transitions 
def mk_gol_tensors(px, timesteps=10, grid_size=9, st_stop=True,
                                                    env=0):
    grid_sts = np.zeros((timesteps,grid_size,grid_size)).astype(int)
    grid_txs = grid_sts[:-1].copy()
    # allocate pattern (upper left corner to allow motion)
    grid_sts[0,:px.dx.shape[0],:px.dx.shape[1]] = px.dx
    # gol transitions
    for ti in range(1,timesteps):
        grid_sts[ti] = gol_tx(grid_sts[ti-1], expand_dy=False)
        grid_txs[ti-1] = grid_sts[ti] - grid_sts[ti-1]
        # stop if state already occurred (remove st, not tx)
        # rm zero layers to account for moving patterns (e.g., glider)
        if st_stop and is_sx_in_dxs(rm_zero_layers(grid_sts[ti]),grid_sts[:ti]):
            # print(f'\nfound state recurrency at t={ti}:\n\n{grid_sts[ti]}\n')
            cpx = get_complexity(grid_txs,px.sx)
            print(f'\n{px.label} complexity = {cpx} (txs={len(np.unique(grid_txs.nonzero()[0]))}/active cells={px.sx.sum()})\n')
            grid_sts[ti] = 0
            return grid_sts,grid_txs
    return grid_sts,grid_txs

# complexity
def get_complexity(txs,sx):
    return len(np.unique(txs.nonzero()[0]))/sx.sum()

# split spikes into ON & OFF and make spike matrices
def mk_spike_trains(grid_txs):
    # make channels
    on_spikes = np.where(grid_txs==1,1,0)
    off_spikes = np.where(grid_txs==-1,1,0)
    # make spike matrices
    on_trains = mk_train_matrix(on_spikes)
    off_trains = mk_train_matrix(off_spikes)
    return on_trains,off_trains

# to turn a spike tensor into a spike matrix (only active cells)
def mk_train_matrix(spikes):
    trains = np.zeros((spikes.nonzero()[0].shape[0],spikes.shape[0]))
    # for each cell of the grid that has changed its state:
    spike_ids = list(set([(i,j) for i,j in zip(*spikes.nonzero()[1:])]))
    # spike_ids is to avoid repetitions
    for ei,(i,j) in enumerate(spike_ids):
        print(i,j)
        trains[ei] = spikes[:,i,j]
    # import pdb; pdb.set_trace()
    return trains

# purpura distance (energy required for the cycle)
def get_vpdist(spt_matrix,spt2=[],q=1):
    # where there are no on/off spikes
    if len(spt_matrix) == 0:
        return np.zeros(1)
    # TODO: define a good whatever for this
    pq = 1/(1*s)
    # make empty spike train for comparison
    if len(spt2) == 0:
        spt2 = SpikeTrain(spt_matrix[0]*0*s, t_stop=spt_matrix.shape[0])
    # victor purpura for each cell/'neuron'/pixel with at least 1 state change
    vp = []
    for spt_arr in spt_matrix:
        spt = SpikeTrain(spt_arr*s, t_stop=spt_arr.shape[0])
        val = vpdist([spt,spt2],pq)[0,1]
        vp.append(val)
    print(f'vp energy value = {sum(vp)}')
    return np.array(vp)


##############
'''
analyze transition for 10 steps
(pxs only blinker, block and glider)
0) save all txs for 10 tsteps
1) 
2) 
'''

def mk_txs(px, timesteps=10,gol_size=(25,25)):
    # container dict
    txs = {}
    # load all px env domains
    dxs = load_dxs(px)
    # to fill domain for future tsteps
    rows = int((gol_size[0] - dxs.shape[1])/2)
    cols = int((gol_size[1] - dxs.shape[2])/2)
    # transitions
    for i in tqdm(range(dxs.shape[0])):
        txs[i] = []
        dx = np.pad(dxs[i],((rows,rows),(cols,cols)))
        txs[i].append(dx.nonzero())
        for _ in range(timesteps):
            dx = gol_tx(dx,expand_dy=False)
            txs[i].append(dx.nonzero())
    save_as(txs, name=f'energy_gol_txs_{px.label}')
# mk_txs(blinker)

# txs = load_data('energy_gol_txs_blinker')
# dx = reconstruct_nonzeros(txs[100])
# gol_animplot([dx])

# save example
# txs_sample = {}
# for i in range(11):
#     txs_sample[i] = txs[i]
# for ri in np.random.randint(0,2**20,size=100):
#     txs_sample[ri] = txs[ri]
# save_as(txs_sample, name=f'energy_gol_txs_{blinker.label}')

txs_sample = load_data('energy_gol_txs_blinker_sample')
ri = np.random.choice(list(txs_sample.keys()))
dx = reconstruct_nonzeros(txs_sample[ri])
gol_animplot(dx, step=3000)


# domains at t=0 and survival ids at t=1
# def get_tx_ids(px, px_search=[blinker,block,glider]):
#     # container dict
#     # ids = dict.fromkeys([px.label for px in px_search],[])
#     ids = {}
#     for ox in px_search:
#         ids[ox.label] = []
#     # load all px env domains
#     dxs = load_dxs(px)
#     for i in tqdm(range(dxs.shape[0])):
#         dx = dxs[i]
#         dy = gol_tx(dx,expand_dy=True)
#         for ox in px_search:
#             if is_px_in_dx(ox,dy,search_borders=True):
#                 ids[ox.label].append(i)
#                 import pdb; pdb.set_trace()
#     save_as(ids,name=f'energy_{px.label}_xy_ids')
#     return ids
# ids = get_tx_ids(blinker)

# check individual cases
# def analyze_txs(px, id=-1,tsteps=10,plot=True):
#     # load all domains & define one to analyze
#     dxs = load_dxs(px.label)
#     xy_ids = load_data(f'energy_{px.label}_xy_ids')
#     id = id if id >= 0 else np.random.choice(xy_ids)
#     dx = dxs[id]
#     # container
#     txs = np.zeros((tsteps,dx.shape[0]+tsteps*2,dx.shape[1]+tsteps*2))
#     # make the transitions
#     for i in range(tsteps):
#         dy = gol_tx(dx,expand_dy=True)











# look for sequential patterns for one pattern/object
def mk_tx_seq(px, timesteps=10):
    # patterns to look for in domains
    oxs = [blinker,block,glider]
    ox_dxs = {}
    for ox in oxs:
        ox_dxs[ox.label] = []
    # all environmental configurations for px
    px_dxs = load_dxs(px)
    # for every env config, do txs
    for ei in tqdm(range(px_dxs.shape[0])):
        dx = px_dxs[ei]
        for ti in range(timesteps):
            # expanded gol tx
            dy = gol_tx(dx)
            # look for patterns
            for ox in oxs:
                # if some variant of px is in domain
                for vx in ox.vxs.sx:
                    if is_sx_in_dx(vx,dy):
                        ox_dxs[ox.label].append([ti,ei])
    save_as(ox_dxs,name=f'energy_txs_ids_{px.label}')
    return ox_dxs
# ids = mk_tx_seq(blinker)

# blinker_ids = load_data(filename='energy_txs_ids_blinker')

def mk_history_data(ids,pxs=[blinker,block,glider]):
    counts = {}
    configs = {}
    # each pattern produces different patterns
    for px in pxs:
        counts[px.label] = defaultdict(int)
        configs[px.label] = defaultdict(list)
        # ix: [t_id, config_id]
        for ix in ids[px.label]:
            counts[px.label][ix[0]] += 1
            configs[px.label][ix[1]].append(ix[0])
    return counts,configs

# blinker_counts,blinker_configs = mk_history_data(blinker_ids)


# sequential transitions
# ox: pattern/object transitioning
# pxs: patterns to look for in codomains
# def mk_obj_txs(obj,pxs, steps=10):
#     # container 
#     evol = {}
#     for px in pxs:
#         evol[px.label] = {}
#         # load transition domains (t=1)
#         ox_dxs = load_codxs(obj)
#         print(f'\n{obj.label} -> {px.label}')
#         for ti in range(steps):
#             print(f'{px.label} domains t={ti+1}: {ox_dxs.shape[0]}')
#             # ids of domains where px was found
#             ids = is_px_in_dxs(px,ox_dxs)
#             ox_dxs = ox_dxs[ids]
#             print(f't={ti+1} (self-sustain) ids: {ox_dxs.shape[0]}')
#             evol[px.label][ti] = ox_dxs
#             ox_dxs = multi_gol_tx(ox_dxs, expand_dy=False)
#     save_as(evol,name='evol_blinker.gol')
#     return evol

# call
#for px in [blinker,pb0,block,glb]:
# for px in [glb,gla]:
# for px in [blinker]:
#     # tensors and complexity
#     sts,txs = mk_gol_tensors(px)
#     px.txs = sts
#     # convert into spikes
#     on,off = mk_spike_trains(txs)
#     px.ons = on
#     px.offs = off
#     # get vp distance
#     on_vps = get_vpdist(on)
#     off_vps = get_vpdist(off)
#     energy = on_vps.sum() + off_vps.sum()
#     # 0: off, 1: on
#     px.ev = np.zeros((2,max(len(off_vps),len(on_vps))))
#     px.ev[0,:off_vps.shape[0]] = off_vps
#     px.ev[1,:on_vps.shape[0]] = on_vps
#     px.e = np.sum(px.ev)
#     print(f'{px.label} vp energy = {energy}\n')
    
