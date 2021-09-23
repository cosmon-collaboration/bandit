import gvar as gv
import numpy as np

data_files = ['data/callat_a12m220XL_test.h5']
reweight_files = []

fit_states = ['pion', 'proton']
bs_seed = 'a12m220XL'

corr_lst = {
    # PROTON
    'proton':{
        'dsets':[
            'spec/proton',
            'spec/proton_np',
            ],
        'weights'  :[0.5, 0.5],
        't_reverse':[False, False],
        'phase'    :[1, 1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,18.5],
        'ylim'     :[0.55,0.759],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :4,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
    # PION
    'pion':{
        'dsets':['spec/piplus'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,32.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :4,
        'T'        :64,
        't_range'  :np.arange(7,31),
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    }
}

priors = gv.BufferDict()
x      = dict()

priors['proton_E_0']  = gv.gvar(0.616, .06)
priors['proton_zS_0'] = gv.gvar(7.5e-4, 2.e-4)
priors['proton_zP_0'] = gv.gvar(2.8e-3, 0.75e-3)

priors['pion_E_0']  = gv.gvar(0.134, .006)
priors['pion_zS_0'] = gv.gvar(0.0815, .02)
priors['pion_zP_0'] = gv.gvar(0.22,  0.022)

for corr in corr_lst:
    for n in range(1,10):
        # use 2 mpi splitting for each dE
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
        for k in ['type', 'T', 'n_state', 't_range', 'eff_ylim']:
            if k in corr_lst[corr]:
                x[state][k] = corr_lst[corr][k]
        x[state]['color'] = corr_lst[corr]['colors'][sp]
        x[state]['snk']   = snk
        x[state]['src']   = corr_lst[corr]['srcs'][0]
