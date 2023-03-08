import gvar as gv
import numpy as np

data_file = 'data/callat_test.h5'

fit_states = ['mres-L','mres-S', 'pion', 'kaon', 
              'proton', 'lambda', 'sigma', 'xi',
              'delta', 'sigma-star', 'xi-star', 'omega']
bs_seed = 'a09m310'
plot_name = 'a09m310'

corr_lst = {
    'pion':{
        'dsets'     :['a09m310/piplus'],
        'xlim'      :[0,48.5],
        'ylim'      :[0.12,0.169],
        'n_state'   :3,
        't_range'   :np.arange(5,48),
    },
    'kaon':{
        'dsets'     :['a09m310/kplus'],
        'xlim'      :[0,48.5],
        'ylim'      :[0.23,0.26],
        'n_state'   :3,
        't_range'   :np.arange(5,48),
    },
    'mres-L':{
        'corr_array':False,
        'dset_MP'   :['a09m310/mp_l'],
        'dset_PP'   :['a09m310/pp_l'],
        'xlim'      :[0,48.5],
        'ylim'      :[0,5e-4],
        't_range'   :np.arange(5,48),
    },
    'mres-S':{
        'corr_array':False,
        'dset_MP'   :['a09m310/mp_s'],
        'dset_PP'   :['a09m310/pp_s'],
        'xlim'      :[0,48.5],
        'ylim'      :[0,5e-4],
        't_range'   :np.arange(5,48),
    },
    'proton':{
            'dsets'    :['a09m310/proton'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.46,0.54],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'lambda':{
            'dsets'    :['a09m310/lambda_z'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.51,0.59],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'sigma':{
            'dsets'    :['a09m310/sigma_p'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.54,0.62],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'xi':{
            'dsets'    :['a09m310/xi_z'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.57,0.65],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'delta':{
            'dsets'    :['a09m310/delta_pp'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.63,0.71],
            'n_state'  :4,
            't_range'  :np.arange(4,15),
        },
    'sigma-star':{
            'dsets'    :['a09m310/sigma_star_p'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.66,0.74],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'xi-star':{
            'dsets'    :['a09m310/xi_star_z'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.69,0.77],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
    'omega':{
            'dsets'    :['a09m310/omega_m'],
            'xlim'     :[0,25.5],
            'ylim'     :[0.73,0.81],
            'n_state'  :4,
            't_range'  :np.arange(4,21),
        },
}

priors = gv.BufferDict()
x      = dict()

priors['pion_E_0']  = gv.gvar(0.14, .006)
priors['pion_zS_0'] = gv.gvar(4.7e-3, 5e-4)
priors['pion_zP_0'] = gv.gvar(0.125,  0.015)

priors['kaon_E_0']  = gv.gvar(0.241, .006)
priors['kaon_zS_0'] = gv.gvar(3.8e-3, 5e-4)
priors['kaon_zP_0'] = gv.gvar(0.1, 0.0125)

priors['mres-L']    = gv.gvar(2.75e-4, 5e-5)
priors['mres-S']    = gv.gvar(2e-4, 5e-5)

priors['proton_E_0']  = gv.gvar(0.495, .02)
priors['proton_zS_0'] = gv.gvar(4.1, 1)
priors['proton_zP_0'] = gv.gvar(4.7, 1)

priors['lambda_E_0']  = gv.gvar(0.54, .02)
priors['lambda_zS_0'] = gv.gvar(4.0, 1)
priors['lambda_zP_0'] = gv.gvar(4.8, 1)

priors['sigma_E_0']  = gv.gvar(0.57, .02)
priors['sigma_zS_0'] = gv.gvar(4.5, 1)
priors['sigma_zP_0'] = gv.gvar(6.0, 1)

priors['xi_E_0']  = gv.gvar(0.602, .02)
priors['xi_zS_0'] = gv.gvar(4.0, 1)
priors['xi_zP_0'] = gv.gvar(5.5, 1)

priors['delta_E_0']  = gv.gvar(0.66, .02)
priors['delta_zS_0'] = gv.gvar(6.3, 2)
priors['delta_zP_0'] = gv.gvar(10, 2)

priors['sigma-star_E_0']  = gv.gvar(0.69, .02)
priors['sigma-star_zS_0'] = gv.gvar(5.8, 1.5)
priors['sigma-star_zP_0'] = gv.gvar(9.5, 2)

priors['xi-star_E_0']  = gv.gvar(0.72, .02)
priors['xi-star_zS_0'] = gv.gvar(5.2, 1.5)
priors['xi-star_zP_0'] = gv.gvar(9.5, 2)

priors['omega_E_0']  = gv.gvar(0.76, .02)
priors['omega_zS_0'] = gv.gvar(5, 1)
priors['omega_zP_0'] = gv.gvar(8, 1)


for corr in corr_lst:
    corr_lst[corr]['weights']   = [1]
    corr_lst[corr]['t_reverse'] = [False]
    if 'mres' in corr:
        corr_lst[corr]['snks']  = ['M', 'P']
        corr_lst[corr]['srcs']  = ['P']
        corr_lst[corr]['colors']= '#51a7f9'
    else:
        corr_lst[corr]['snks']  = ['S', 'P']
        corr_lst[corr]['srcs']  = ['S']
        corr_lst[corr]['colors']= {'SS':'#51a7f9','PS':'k'}
        corr_lst[corr]['ztype'] = 'z_snk z_src'
    if corr in ['pion', 'kaon'] or 'mres' in corr:
        corr_lst[corr]['fold']  = True
        corr_lst[corr]['T']     = 96
        if 'mres' in corr:
            corr_lst[corr]['type'] = 'mres'
        else:
            corr_lst[corr]['type'] = 'cosh'
            corr_lst[corr]['t_sweep'] = range(2,28)
            corr_lst[corr]['n_sweep'] = range(1,6)
    else:
        corr_lst[corr]['fold']      = False
        corr_lst[corr]['type']      = 'exp'
        corr_lst[corr]['t_sweep']   = range(2,16)
        corr_lst[corr]['normalize'] = True


for corr in corr_lst:#[k for k in corr_lst if 'mres' not in k]:
    if 'mres' not in corr:
        for n in range(1,10):
            # use 2 mpi splitting for each dE

            # E_n = E_0 + dE_10 + dE_21 +...
            # use log prior to force ordering of dE_n
            priors['log(%s_dE_%d)' %(corr,n)] = gv.gvar(np.log(2*priors['pion_E_0'].mean), 0.7)

            # for z_P, no suppression with n, but for S, smaller overlaps
            zP_0 = priors['%s_zP_0' %(corr)]
            priors['%s_zP_%d' %(corr,n)] = gv.gvar(zP_0.mean, 3*zP_0.mean)

            zS_tag = 'S'
            zS_0 = priors['%s_z%s_0' %(corr, zS_tag)]
            if n <= 2:
                priors['%s_z%s_%d' %(corr, zS_tag, n)] = gv.gvar(zS_0.mean, 2*zS_0.mean)
            else:
                priors['%s_z%s_%d' %(corr, zS_tag, n)] = gv.gvar(0, zS_0.mean)
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
