import gvar as gv
import numpy as np

data_file = './data/callat_a09m310_test.h5'

fit_states = ['mres-L','mres-S', 'pion', 'kaon', 'proton', 'omega']
#fit_states = ['pion', 'kaon', 'proton', 'omega']
bs_seed = 'a12m220XL'

corr_lst = {
    # PION
    'pion':{
        'dsets':['a09m310/piplus'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,48.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :3,
        'T'        :96,
        't_range'  :np.arange(5,48),
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    },
    # KAON
    'kaon':{
        'dsets':['a09m310/kplus'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,48.5],
        'ylim'     :[0.233,0.255],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :4,
        'T'        :96,
        't_range'  :np.arange(8,48),
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    },
    # MRES_L
    'mres-L':{
        'corr_array':False,
        'dset_MP'   :['a09m310/mres-L_MP'],
        'dset_PP'   :['a09m310/mres-L_PP'],
        'weights'   :[1],
        't_reverse' :[False],
        'fold'      :True,
        'T'         :96,
        'snks'      :['M', 'P'],
        'srcs'      :['P'],
        'xlim'      :[0,48.5],
        'ylim'      :[2e-4,4e-4],
        'colors'    :'#70bf41',
        'type'      :'mres',
        # fit params
        't_range'   :np.arange(10,49),
        't_sweep'   :range(2,28),
    },
    # MRES_S
    'mres-S':{
        'corr_array':False,
        'dset_MP'   :['a09m310/mres-S_MP'],
        'dset_PP'   :['a09m310/mres-S_PP'],
        'weights'   :[1],
        't_reverse' :[False],
        'fold'      :True,
        'T'         :96,
        'snks'      :['M', 'P'],
        'srcs'      :['P'],
        'xlim'      :[0,48.5],
        'ylim'      :[1.25e-4,3.25e-4],
        'colors'    :'#70bf41',
        'type'      :'mres',
        # fit params
        't_range'   :np.arange(10,49),
        't_sweep'   :range(2,28),
    },
    # PROTON
    'proton':{
        'dsets':['a09m310/proton'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,25.5],
        'ylim'     :[0.425,0.575],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
    # OMEGA
    'omega':{
        'dsets':['a09m310/omega_m'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,28.5],
        'ylim'     :[0.675,0.825],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        #'z_ylim'   :[0.,5],
        #'normalize':True,
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(10,23),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },

}

priors = gv.BufferDict()
x      = dict()

priors['proton_E_0']  = gv.gvar(0.5, .06)
priors['proton_zS_0'] = gv.gvar(2.0e-5, 1.e-5)
priors['proton_zP_0'] = gv.gvar(2.5e-3, 1.e-3)

priors['omega_E_0']  = gv.gvar(0.75, .06)
priors['omega_zS_0'] = gv.gvar(3e-5, 1.e-5)
priors['omega_zP_0'] = gv.gvar(.0025, .001)

priors['pion_E_0']  = gv.gvar(0.14, .006)
priors['pion_zS_0'] = gv.gvar(5e-3, 5e-4)
priors['pion_zP_0'] = gv.gvar(0.125,  0.015)

priors['kaon_E_0']  = gv.gvar(0.241, .006)
priors['kaon_zS_0'] = gv.gvar(4e-3, 5e-5)
priors['kaon_zP_0'] = gv.gvar(0.1, 0.025)

priors['mres-L']    = gv.gvar(2.75e-4, 5e-5)
priors['mres-S']    = gv.gvar(2e-4, 5e-5)

for corr in corr_lst:#[k for k in corr_lst if 'mres' not in k]:
    if 'mres' not in corr:
        for n in range(1,10):
            # use 2 mpi splitting for each dE

            # E_n = E_0 + dE_10 + dE_21 +...
            # use log prior to force ordering of dE_n
            priors['log(%s_dE_%d)' %(corr,n)] = gv.gvar(np.log(2*priors['pion_E_0'].mean), 0.7)

            # for z_P, no suppression with n, but for S, smaller overlaps
            priors['%s_zP_%d' %(corr,n)] = gv.gvar(priors['%s_zP_0' %(corr)].mean, 2*priors['%s_zP_0' %(corr)].sdev)
            zS_0 = priors['%s_zS_0' %(corr)]
            if n <= 2:
                priors['%s_zS_%d' %(corr,n)] = gv.gvar(zS_0.mean, 2*zS_0.sdev)
            else:
                priors['%s_zS_%d' %(corr,n)] = gv.gvar(zS_0.mean/2, zS_0.sdev)
    # x-params
    for snk in corr_lst[corr]['snks']:
        sp = snk+corr_lst[corr]['srcs'][0]
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
