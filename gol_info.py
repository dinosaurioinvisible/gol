from gol_info_utils import *
import pandas as pd


'''
agency

1) mk all domains (dxs)
2) mk all codomains (dxys)
3) find py patterns in px codomains (structural transitions)
4) build transition map
5) examine tx map, output tx table
'''

# 1)
# mk_px_domains(pb0, cap=False, save=True)
# 2)
# mk_px_codomains(pb0, cap=False, save=True)
# 3)
# mk_px_data(pxs)
# 4) 
# mk_tx_map(pxs)
# 5) 
# txmap = load_data(filename='gol_txmap_no_cap')
# txs = get_tx_counts(txmap,as_df=False,to_csv=True)

ppxs = [blinker,pb0,block,gla,glb,flag,ttt,ttl,worm,boat]
plot_pxs(ppxs, title='GoL patterns')


# just to double check

# dk = {}
# bk_dxys = load_codxs(blinker)
# for py in pxs:
#     dk[py.label] = is_px_in_dxs(py,bk_dxys)
#     print(f'{py.label} cases = {dk[py.label].nonzero()[0].shape[0]}')


# search for same envs within different env sets
def get_exs_in_eks(txmap,txs, return_ids=False,to_csv=False):
    txs_envs = {}
    txs_envs_ids = {}
    # real sx ->
    for px in txs.keys():
        txs_envs[px] = {}
        if return_ids:
            txs_envs_ids = {}
        print(f'\n\npx: {px}')
        # sx -> sy
        for py in txs.keys():
            print(f'\npx: {px} -> py: {py}')
            xy_ids = set(txmap[px][py]['ids'])
            # if sx -> sy
            if len(xy_ids) > 0:
                txs_envs[px][py] = {}
                if return_ids:
                    txs_envs_ids[px][py] = {}
                # all alternative sxs -> sy
                sx_exs = 0
                for altx in txs.keys():
                    exs = xy_ids.intersection(txmap[altx][py]['ids'])
                    txs_envs[px][py][altx] = len(exs)
                    if return_ids:
                        txs_envs_ids[px][py][altx] = exs
                    sx_exs += len(exs)
                    print(f'{px}({altx}) -> {py}: {len(exs)}/{len(xy_ids)}')
                print(f'total alternative sx-exs: {sx_exs}')
        if to_csv:
            px_py_envs = pd.DataFrame.from_dict(txs_envs[px])
            px_py_envs.to_csv(f'alt_sx={px}.csv')
    if return_ids:
        return txs_envs_ids
    return txs_envs
# get_exs_in_eks(txmap,txs, to_csv=True)

