
from gol_info_utils import *
import numpy as np
# victor purpura stuff
from quantities import s
from neo.core import SpikeTrain
from elephant.spike_train_dissimilarity import victor_purpura_distance as vpdist


'''
pipeline:
1) make tensors and convert to spikes
2) complexity score value
3) victor purpura energy measures
    # define q
4) environmental pressure
    # work this out
5)
'''

# px: pattern (blinker,block,etc), grid (states) & grid txs tensors
# 11 timesteps, so 11 grid states and 10 grid transitions 
def mk_gol_tensors(px, timesteps=11, grid_size=9, st_stop=True,
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

# call
#for px in [blinker,pb0,block,glb]:
for px in [glb,gla]:
    # tensors and complexity
    sts,txs = mk_gol_tensors(px)
    px.txs = sts
    # convert into spikes
    on,off = mk_spike_trains(txs)
    px.ons = on
    px.offs = off
    # get vp distance
    on_vps = get_vpdist(on)
    off_vps = get_vpdist(off)
    energy = on_vps.sum() + off_vps.sum()
    # 0: off, 1: on
    px.ev = np.zeros((2,max(len(off_vps),len(on_vps))))
    px.ev[0,:off_vps.shape[0]] = off_vps
    px.ev[1,:on_vps.shape[0]] = on_vps
    px.e = np.sum(px.ev)
    print(f'{px.label} vp energy = {energy}\n')
    
    