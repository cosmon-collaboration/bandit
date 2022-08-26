import numpy as np
import gvar as gv
import lsqfit
import copy
import sys
#
import fitter.corr_functions as cf
import fitter.load_data as ld
import fitter.plotting as plot


def run_stability(states, xp, x, y, gv_data, data_cfg,
                  es_stability=False, svd_test=True,
                  save_figs=False, scale=[]):

    for state in states:
        if 't_sweep' in xp.corr_lst[state]:
            tmin = xp.corr_lst[state]['t_sweep']
        else:
            tmin = range(2, x[state]['t_range'][-1])
        if 'n_sweep' in xp.corr_lst[state]:
            n_states = xp.corr_lst[state]['n_sweep']
        else:
            n_states = range(1, 6)
        tn_opt = (xp.corr_lst[state]['t_range'][0],
                  xp.corr_lst[state]['n_state'])

        fits = {}
        for ti in tmin:
            x_tmp = copy.deepcopy(xp.x)
            x_tmp = {k:v for (k,v) in x_tmp.items() if state in k}
            for k in x_tmp:
                x_tmp[k]['t_range'] = np.arange(ti, xp.x[k]['t_range'][-1]+1)

            y_tmp = {k: v[x_tmp[k]['t_range']]
                 for (k, v) in gv_data.items() if k.split('_')[0] in [state]}

            p_tmp = copy.deepcopy(xp.priors)
            for k in xp.priors:
                if state not in k:
                    p_tmp.pop(k)

            if svd_test:
                has_svd = True
                svd_test, svdcut = ld.svd_diagnose(y_tmp, data_cfg, x_tmp)
                has_svd = True

            if ti == tmin[0]:
                print([k for k in y_tmp])
            for ns in n_states:
                xx = copy.deepcopy(x_tmp)
                for k in xx:
                    ''' NOTE  - we are chaning n_s for pi, D and Dpi all together '''
                    xx[k]['n_state'] = ns
                fit_funcs = cf.FitCorr()
                p_sweep = {}
                for k in p_tmp:
                    if int(k.split('_')[-1].split(')')[0]) < ns:
                        p_sweep[k] = p_tmp[k]
                p0 = {k: v.mean for (k, v) in p_tmp.items()}
                sys.stdout.write('sweeping t_min = %d n_s = %d\r' % (ti, ns))
                sys.stdout.flush()
                if has_svd:
                    f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
                                                 prior=p_sweep, p0=p0,
                                                 fcn=fit_funcs.fit_function, svdcut=svdcut)
                else:
                    f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
                                                 prior=p_sweep, p0=p0,
                                                 fcn=fit_funcs.fit_function)
                fits[(ti, ns)] = f_tmp

        ylim = None

        plot.plot_stability(fits, tmin, n_states, tn_opt,
                            state, ylim=ylim, save=save_figs, scale=scale)
        if es_stability:
            for i_n in range(1, n_states[-1]):
                plot.plot_stability(fits, tmin, n_states, tn_opt, state,
                                    ylim=ylim, save=save_figs, n_plot=i_n, scale=scale)
