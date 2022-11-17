import gvar as gv
import numpy as np

data_file = 'data/callat_test.h5'

fit_states = ['proton', 'pion']
#fit_states = ['pion', 'kaon', 'proton', 'omega']
bs_seed = 'a09m310'
plot_name = 'a09m310'

corr_lst = {
    # PION
    'pion':{
        'dsets':['a09m310_new_old/piplus'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'srcs'     :['S1', 'S2'],
        'snks'     :[['S1', 'P'], ['S2', 'P']],
        'xlim'     :[0,48.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'S1S1':'#70bf41','PS1':'k', 'S2S2':'b','PS2':'gray'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        'eff_ylim' :[0.133,0.1349],
        # optimal fit params
        'n_state'  :3,
        'T'        :96,
        't_range'  :np.arange(5,48),
        # stability fit parameters
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
    },
    # PROTON
    'proton':{
        'dsets':['a09m310_new_old/proton'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'srcs'     :['S1', 'S2'],
        'snks'     :[['S1', 'P'], ['S2', 'P']],
        'xlim'     :[0,25.5],
        'ylim'     :[0.425,0.575],
        'colors'   :{'S1S1':'#70bf41','PS1':'k', 'S2S2':'b','PS2':'gray'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
}

priors = gv.BufferDict()
x      = dict()

priors['proton_E_0']  = gv.gvar(0.495, .025)
priors['proton_zS2_0'] = gv.gvar(2.2e-5, 0.5e-5)
priors['proton_zS1_0'] = gv.gvar(3.4e-4, 0.5e-4)
priors['proton_zP_0'] = gv.gvar(1.2e-3, 2.5e-4)

priors['pion_E_0']  = gv.gvar(0.14, .006)
priors['pion_zS1_0'] = gv.gvar(4.7e-3, 5e-4)
priors['pion_zS2_0'] = gv.gvar(4.7e-5, 5e-6)
priors['pion_zP_0'] = gv.gvar(0.125,  0.015)

for corr in corr_lst:#[k for k in corr_lst if 'mres' not in k]:
    if 'mres' not in corr:
        for n in range(1,10):
            # use 2 mpi splitting for each dE

            # E_n = E_0 + dE_10 + dE_21 +...
            # use log prior to force ordering of dE_n
            priors['log(%s_dE_%d)' %(corr,n)] = gv.gvar(np.log(2*priors['pion_E_0'].mean), 0.7)

            # for z_P, no suppression with n, but for S, smaller overlaps
            priors['%s_zP_%d' %(corr,n)] = gv.gvar(priors['%s_zP_0' %(corr)].mean, priors['%s_zP_0' %(corr)].mean)
            zS1_0 = priors['%s_zS1_0' %(corr)]
            zS2_0 = priors['%s_zS1_0' %(corr)]
            if n <= 2:
                priors['%s_zS1_%d' %(corr, n)] = gv.gvar(zS1_0.mean, zS1_0.mean)
                priors['%s_zS2_%d' %(corr, n)] = gv.gvar(zS2_0.mean, zS2_0.mean)
            else:
                priors['%s_zS1_%d' %(corr, n)] = gv.gvar(zS1_0.mean/2, zS1_0.mean/2)
                priors['%s_zS2_%d' %(corr, n)] = gv.gvar(zS2_0.mean/2, zS2_0.mean/2)

    # x-params
    for j, src in enumerate(corr_lst[corr]['srcs']):
        for snk in corr_lst[corr]['snks'][j]:
            sp = snk+src
            state = corr+'_'+sp
            x[state] = dict()
            x[state]['state'] = corr
            for k in ['type', 'T', 'n_state', 't_range', 'eff_ylim', 'ztype']:
                if k in corr_lst[corr]:
                    x[state][k] = corr_lst[corr][k]
            if 't0' in corr_lst[corr]:
                x[state]['t0'] = corr_lst[corr]['t0']
            else:
                x[state]['t0'] = 0
            if 'mres' not in corr:
                x[state]['color'] = corr_lst[corr]['colors'][sp]
                x[state]['snk']   = snk
                x[state]['src']   = corr_lst[corr]['srcs'][0]
            else:
                x[state]['color'] = corr_lst[corr]['colors']
