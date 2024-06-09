

# # dx/dy ids for px -> py transition
# # search patterns in codomains (i.e., find transitions)
# def find_pxpy_txs_v0(px,pxs, cap=False,dx_constraint=True,save=True):
#     from copy import deepcopy
#     pxs_cp = deepcopy(pxs)
#     cap = cap if cap else f'no_cap={px.env.sum()}'
#     codomain_fname = f'gol_tx_domains_cap={cap}_{px.label}'
#     px_dys = load_data(filename=codomain_fname)
#     # to search in the whole codomain matrix, or omit the borders
#     di,dj = px_dys.shape[1:]
#     vi,vj = 0,0
#     px_txs = {}
#     for py in pxs_cp:
#         print(f'\n{px.label} -> {py.label:}')
#         ids = np.zeros(px_dys.shape[0]).astype(int)
#         for i in tqdm(range(py.vxs.sx.shape[0])):
#             sy = py.vxs[i].sx
#             # patterns outside dx space are made from env
#             if dx_constraint:
#                 vi,vj = (np.array(px_dys.shape[1:] - np.array(sy.shape) -2)/2).astype(int)
#             ids += is_sx_in_dxs(sy,px_dys[:,vi:di-vi,vj:dj-vj])
#         # is there more than 1 sx in dx?
#         if ids[ids>1].shape[0] > 0:
#             print(f'\nmore than one sx?:\n{ids[ids>1].shape[0]} cases\n{ids[ids>1]}')
#             examples = np.where(ids>1)[0][:5]
#             for exi in examples:
#                 print(f'\n{exi}')
#                 print(px_dys[exi])
#             # import pdb; pdb.set_trace()
#         px_txs[py.label] = np.where(ids>0)[0]
#     # disintegrations
#     end_txs = np.where(np.sum(px_dys[:,2:-2,2:-2]*px.sx,axis=(1,2))<1)[0]
#     if end_txs > 0:
#         px_txs['end']
#     # info
#     all_txs = px_dys.shape[0]
#     ntxs = 0
#     print()
#     for key in px_txs.keys():
#         ntxs += px_txs[key].shape[0]
#         print(f'{key}: {px_txs[key].shape[0]}')
#     print(f'{ntxs} txs known, {all_txs - ntxs}/{all_txs} unknown')
#     if save:
#         fname = 'gol_px_txs_{}cap={}_{}'.format('ct_' if dx_constraint else '', cap,px.label)
#         save_as(px_txs,name=fname)
#     else:
#         return px_txs

# def mk_omap(pxs):
#     omap = {}
#     labels = []
#     for px in pxs:
#         labels.append(px.label)
#         omap[px.label] = {}
#         omap[px.label]['fwd'] = {}
#         omap[px.label]['back'] = {}
#         omap[px.label]['nfwd'] = {}
#         omap[px.label]['nback'] = {}
#         omap[px.label]['edx'] = {}
#     print()
#     for px in pxs:
#         for li in tqdm(range(len(labels))):
#             py_label = labels[li]
#             omap[px.label]['fwd'][py_label] = px.txs[py_label]
#             omap[py_label]['back'][px.label] = px.txs[py_label]
#             omap[px.label]['nfwd'][py_label] = px.txs[py_label].shape[0]
#             omap[py_label]['nback'][px.label] = px.txs[py_label].shape[0]
#             omap[px.label]['edx'][py_label] = px.env_sets[py_label]
#     return omap

# sx -> sy : tx ids : env set
# sx -> sy : n txs : env set weight/cardinality
# sx <- sy : tx ids : backwards env set?
# sx <- sy : n txs : backwards weight/cardinality
# sx -> sy : dom x
# sx -> sy : dom y
# sx -> sy : delta sx,sy
# sx -> sy : eks : environmental distinctions
# sx -> sy : dom y sums
# def mk_delta_map(pxs,omap):
#     # make dicts
#     dmap = {}
#     dmap['labels'] = list(omap.keys())
#     for px in pxs:
#         dmap[px.label] = {}
#         for ku in ['fwd', 'nfwd', 'back', 'nback', 'env_dks', 'dxs', 'dys', 'sxy', 'dysums']:
#             dmap[px.label][ku] = {}
#     # direct copy from omap
#     for px in pxs:
#         for py_label in dmap['labels']:
#             # fwd tx ids, n, bwd tx ids, n, env distinctions
#             for txu in ['fwd', 'nfwd', 'back', 'nback']:
#                 dmap[px.label][txu][py_label] = omap[px.label][txu][py_label]
#                 dmap[px.label]['env_dks'][py_label] = omap[px.label]['edx'][py_label]
#     # load domains data
#     for px in pxs:
#         px_dxs = load_data(filename=f'gol_domains_cap=10_{px.label}')
#         px_dys = load_data(filename=f'gol_tx_domains_cap=10_{px.label}')
#         px_dys_ct = px_dys[:,1:-1,1:-1]
#         pxdx = px.dx.astype(int)
#         for py_label in dmap['labels']:
#             # sx domains, sy domains, sxy transition, sy domain sums 
#             dxs_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0],pxdx.shape[1])).astype(int)
#             dys_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0]+2,pxdx.shape[1]+2)).astype(int)
#             sxy_tensor = np.zeros((omap[px.label]['nfwd'][py_label],pxdx.shape[0],pxdx.shape[1])).astype(int)
#             for ei,id in enumerate(omap[px.label]['fwd'][py_label]):
#                 dxs_tensor[ei] = px_dxs[id]
#                 dys_tensor[ei] = px_dys[id]
#                 sxy_tensor[ei] = pxdx + px_dys_ct[id]
#             dmap[px.label]['dxs'][py_label] = dxs_tensor
#             dmap[px.label]['dys'][py_label] = dys_tensor
#             dmap[px.label]['sxy'][py_label] = sxy_tensor
#             dmap[px.label]['dysums'] = dys_tensor.sum(axis=0)[1:-1,1:-1]
#     fname = f'gol_delta_map_cap=10'
#     save_as(dmap, name=fname)
#     # return dmap