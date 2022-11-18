import numpy as np
import gvar as gv
import lsqfit
import tqdm
import h5py
import copy
import sys
import os
#
import fitter.corr_functions as cf
import fitter.load_data as ld
import fitter.plotting as plot


def run_stability(args, xp, x, y, gv_data, data_cfg, plot_name):
    states       = args.stability
    es_stability = args.es_stability
    svd_test     = args.svd_test
    save_figs    = args.save_figs
    scale        = args.scale

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

            if args.svd_test:
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

        plot.plot_stability(fits, tmin, n_states, tn_opt, state,
                            ylim=ylim, save=save_figs, scale=scale, plot_name=plot_name)
        if es_stability:
            for i_n in range(1, n_states[-1]):
                plot.plot_stability(fits, tmin, n_states, tn_opt, state,
                                    ylim=ylim, save=save_figs, n_plot=i_n,
                                    scale=scale, plot_name=plot_name)


def run_bootstrap(args, fit, fp, data_cfg, x_fit, svdcut=None):

    # make sure results dir exists
    if args.bs_write:
        if not os.path.exists('bs_results'):
            os.makedirs('bs_results')
    if len(args.bs_results.split('/')) == 1:
        bs_file = 'bs_results/'+args.bs_results
    else:
        bs_file = args.bs_results

    # check if we already wrote this dataset
    if args.bs_write:
        have_bs = False
        if os.path.exists(bs_file):
            with h5py.File(bs_file, 'r') as f5:
                if args.bs_path in f5:
                    if len(f5[args.bs_path]) > 0 and not args.overwrite:
                        have_bs = True
                        print(
                            'you asked to write bs results to an existing dset and overwrite =', args.overwrite)
    else:
        have_bs = False

    if not have_bs:
        print('beginning Nbs=%d bootstrap fits' % args.Nbs)
        import fitter.bs_utils as bs
        fit_funcs = cf.FitCorr()

        # use the fit posterior to set the initial guess for bs loop
        p0_bs = dict()
        for k in fit.p:
            p0_bs[k] = fit.p[k].mean

        # seed bs random number generator
        if not args.bs_seed and 'bs_seed' not in dir(fp):
            tmp = input('you have not passed a BS seed nor is it defined in the input file\nenter a seed or hit return for none')
            if not tmp:
                bs_seed = None
            else:
                bs_seed = tmp
        elif 'bs_seed' in dir(fp):
            bs_seed = fp.bs_seed
        if args.bs_seed:
            if args.verbose:
                print('WARNING: you are overwriting the bs_seed from the input file')
            bs_seed = args.bs_seed

        # create bs_list
        Ncfg = data_cfg[next(iter(data_cfg))].shape[0]
        bs_list = bs.get_bs_list(Ncfg, args.Nbs, Mbs=args.Mbs, seed=bs_seed)

        # make BS list for priors
        p_bs_mean = dict()
        for k in fit.prior:
            if args.bs0_restrict and '_0' in k:
                sdev = args.bs0_width * fit.p[k].sdev
            else:
                sdev = fit.prior[k].sdev
            if 'log' in k:
                dist_type='lognormal'
            else:
                dist_type='normal'
            p_bs_mean[k] = bs.bs_prior(args.Nbs, mean=fit.prior[k].mean,
                                       sdev=sdev, seed=bs_seed+'_'+k, dist=dist_type)

        # set up posterior lists of bs results
        post_bs = dict()
        for k in fit.p:
            post_bs[k] = []

        fit_str = []

        # loop over bootstraps
        for bs in tqdm.tqdm(range(args.Nbs)):
            ''' all gvar's created in this switch are destroyed at restore_gvar
                [they are out of scope] '''
            gv.switch_gvar()

            bs_data = dict()
            for k in fit.y:
                bs_data[k] = data_cfg[k][bs_list[bs]]
            bs_gv = gv.dataset.avg_data(bs_data)

            if any(['mres' in k for k in bs_gv]):
                bs_tmp = {k:v for (k,v) in bs_gv.items() if 'mres' not in k}
                for k in [key for key in bs_gv if 'mres' in key]:
                    mres = k.split('_')[0]
                    if mres not in bs_tmp:
                        bs_tmp[mres] = bs_gv[mres+'_MP'] / bs_gv[mres+'_PP']
                bs_gv = bs_tmp

            y_bs = {k: v[x_fit[k]['t_range']]
                    for (k, v) in bs_gv.items() if k in fit.y}

            p_bs = dict()
            for k in p_bs_mean:
                p_bs[k] = gv.gvar(p_bs_mean[k][bs], fit.prior[k].sdev)

            if svdcut:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                              fcn=fit_funcs.fit_function, svdcut=svdcut)
            else:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                              fcn=fit_funcs.fit_function)
            fit_str.append(str(fit_bs))
            for r in post_bs:
                post_bs[r].append(fit_bs.p[r].mean)

            ''' end of gvar scope used for bootstrap '''
            gv.restore_gvar()

        for r in post_bs:
            post_bs[r] = np.array(post_bs[r])
        if args.bs_write:
            # write the results
            with h5py.File(bs_file, 'a') as f5:
                try:
                    f5.create_group(args.bs_path)
                except Exception as e:
                    print(e)
                for r in post_bs:
                    if len(post_bs[r]) > 0:
                        if r in f5[args.bs_path]:
                            del f5[args.bs_path+'/'+r]
                        f5.create_dataset(args.bs_path+'/'+r, data=post_bs[r])

        return post_bs, fit_str
    else:
        sys.exit('not running BS - bs_results exist')
