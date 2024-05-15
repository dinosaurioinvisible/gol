
import numpy as np
from gol_auxs import mk_gol_pattern, mk_moore_nb, mk_sx_variants, array2int
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
            self.vxs[ei].id = array2int(vxi)
        
        self.nsxs = len(vxs)
        

blinker = GolPattern('blinker')
pb0 = GolPattern('pb0')
block = GolPattern('block')
gla = GolPattern('gliderA')
glb = GolPattern('gliderB')
tz = GolPattern('tetrisZ')
ttt = GolPattern('tetrisT')
ttl = GolPattern('tetrisL')
zz = GolPattern('zigzag')
bar = GolPattern('bar')
baby = GolPattern('baby')
flag = GolPattern('flag')
kyte = GolPattern('kyte')
worm = GolPattern('worm')