from gol_info_utils import *
# from gol_fxs import *
# from scipy.special import kl_div
# import pandas as pd

# compare exs vs Exs





# kl divergence
def compare_kldiv(px,py_label):
    txs = px.txs[py_label]
    dxs_fname = f'gol_domains_cap=10_{px.label}'
    dxs = load_data(filename=dxs_fname)

    env_set = px.env_sets[py_label].flatten()
    env_ids = px.env.flatten().nonzero()[0]
    env_ct = env_set[env_ids]
    kl_infos = []
    info_acum = 0
    print(f'\n{px.label} -> {py_label}')
    for ei,tx_id in enumerate(txs):
        print(f'{ei} - id: {tx_id}')

        env = dxs[tx_id].flatten()[env_ids]
        print(f'env: {env}')
        print(f'env cat: {env_ct}\n')

        kld_info = kl_div(env,env_ct)
        print(f'scipy: {kld_info}')
        print('KLD(P||Q): P=env, Q=cog. Info lost when approx using Q instead of P')
        kld_info_pq = np.sum(np.where(env!=0, env*np.log(env/env_ct), 0))
        print(f'KLD Q=cog info: {kld_info_pq}')
        info_acum += kld_info_pq
        kl_infos.append(kld_info_pq)
        print('KLD(Q|P): ?')
        kld_info_qp = np.sum(np.where(env_ct!=0, env_ct*np.log(env_ct/env), 0))
        print(f'KLD Q=env info: {kld_info_qp}')

        print(f'acum. info: {info_acum}\n')
        import pdb; pdb.set_trace()

    print(f'\nacum. info: {info_acum}\n')
    return np.array(kl_infos)




