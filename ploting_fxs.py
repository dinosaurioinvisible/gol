
# ploting fxs for the gol experiments

from gol_auxs import *
from gol_fxs import *
from gol_info_utils import *
# from gol_info import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

'''
protocog agency stuff
1) get patterns
2) check domains and codomains forward
3) check domains and codomains backwards
4) make transitions table
5) get information results
6) make transition diagrams and plots
7) check and save
'''

# 1) get patterns - imported from gol_info_utils 
# basic
blinker = GolPattern('blinker')
pb0 = GolPattern('pb0')
block = GolPattern('block')
gla = GolPattern('gliderA')
glb = GolPattern('gliderB')
patterns = [blinker, pb0, block, gla, glb]
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
patterns += [ttz, ttt, ttl, zz, bar, baby, flag, kite, worm]
# new, after alife24
tub = GolPattern('tub')
helix = GolPattern('helix')
boat = GolPattern('boat')
pbar = GolPattern('prybar')                       # 128 variants and env=24
patterns += [tub, helix, boat, pbar]
# these have 2048 variants each
ffly = GolPattern('firefly')
ufo = GolPattern('ufo')
glider = append_vxs('glider',gla,glb)
patterns += [ffly, ufo, glider]

pxs = []
for px in patterns:
    fname = f'alt_sx={px.label}.csv'
    if os.path.isfile(fname):
        px.txs = pd.read_csv(fname)
        px.txs.rename(columns={'Unnamed 0' : 'pattern'})
        pxs.append(px)


# 2)