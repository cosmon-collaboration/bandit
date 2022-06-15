import fitter.plotting as plot
import fitter.corr_functions as cf
import fitter.load_data as ld
#import fitter.fit_analyzer as analyze
import lsqfit
import gvar as gv
import importlib
import h5py
import copy
import os
import sys
import pathlib
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np
np.seterr(invalid='ignore')
#import tables as h5


# user libs
#sys.path.append('util')


def main():
    parser = argparse.ArgumentParser(
        description='Perform analysis of two-point correlation function')
    parser.add_argument('fit_params',    help='input file to specify fit')
    parser.add_argument('--fit',         default=False, action='store_true',
                        help=            'do fit? [%(default)s]')
    parser.add_argument('--svdcut',      type=float, help='add svdcut to fit')
    parser.add_argument('--svd_test',    default=True, action='store_false',
                        help=            'perform gvar svd_diagnosis? [%(default)s]')
    parser.add_argument('--svd_nbs',     type=int, default=50, help='number of BS samples for estimating SVD cut [%(default)s]')
    parser.add_argument('--fold',        default=True, action='store_false',
                        help=            'fold data about T/2? [%(default)s]')
    parser.add_argument('-b', '--block', default=1, type=int,
                        help=            'specify bin/blocking size in terms of saved configs')
    parser.add_argument('--eff',         default=False, action='store_true',
                        help=            'plot effective mass and z_eff data? [%(default)s]')
    parser.add_argument('--scale',       default=None, nargs=2,
                        help=            'add right axis with physical scale specified by input value [scale, units]')
    parser.add_argument('--stability',   nargs='+',
                        help=            'specify states to perform t_min and n_state sweep')
    parser.add_argument('--es_stability', default=False, action='store_true',
                        help=            'plot excited state stability? [%(default)s]')
    parser.add_argument('--states',      nargs='+',
                        help=            'specify states to fit?')
    parser.add_argument('-v','--verbose', default=False,action='store_true',
                        help=            'add verbosity [%(default)s]')
    parser.add_argument('--verbose_fit', default=False, action='store_true',
                        help=            'print y vs f(x,p) also? [%(default)s]')
    parser.add_argument('--save_figs',   default=False, action='store_true',
                        help=            'save figs? [%(default)s]')
    parser.add_argument('--bs',          default=False, action='store_true',
                        help=            'run bootstrap fit? [%(default)s]')
    parser.add_argument('--Nbs',         type=int, default=2000,
                        help=            'specify the number of BS samples to compute [%(default)s]')
    parser.add_argument('--bs_seed',     default=None,
                        help=            'set a string to seed the bootstrap - None will be random [%(default)s]')
    parser.add_argument('--bs_results',  default='bs_results/spectrum_bs.h5',
                        help=            'set file to write bootstrap results [%(default)s]')
    parser.add_argument('--overwrite',   default=False, action='store_true',
                        help=            'overwrite existing bootstrap results? [%(default)s]')
    parser.add_argument('--bs_path',     default='spec',
                        help=            'specify path in h5 file for bs results')
    parser.add_argument('--uncorr_corrs', default=False, action='store_true',
                        help=            'uncorrelate different correlation functions? [%(default)s]')
    parser.add_argument('--uncorr_all',  default=False, action='store_true',
                        help=            'uncorrelate all snk,src for each correlation function? [%(default)s]')
    parser.add_argument('--interact',    default=False, action='store_true',
                        help=            'open IPython instance after to interact with results? [%(default)s]')
    parser.add_argument('--gui',    default=False, action='store_true',
                        help=            'open dashboard for analyzing fit. Must be used together with fit flag. [%(default)s]')

    args = parser.parse_args()
    if args.save_figs and not os.path.exists('figures'):
        os.makedirs('figures')
    print(args)
    # add path to the input file and load it
    sys.path.append(os.path.dirname(os.path.abspath(args.fit_params)))
    fp = importlib.import_module(
        args.fit_params.split('/')[-1].split('.py')[0])

    # can only uncorrelate all or sets of corrs
    if args.uncorr_all and args.uncorr_corrs:
        sys.exit('you can only select uncorr_corrs or uncorr_all')
    # re-weight correlators?
    try:
        reweight = fp.reweight
    except:
        reweight = False
    # block data
    bl = args.block
    if 'block' in dir(fp) and fp.block != 1:
        bl = fp.block
    if args.block != 1: # allow cl override
        bl = args.block
    if reweight:
        rw_files = fp.rw_files
        rw_path = fp.rw_path
        gv_data = ld.load_h5(fp.data_file, fp.corr_lst, rw=[rw_files, rw_path], bl=bl,
                             uncorr_corrs=args.uncorr_corrs, uncorr_all=args.uncorr_all)
        data_cfg = ld.load_h5(fp.data_file, fp.corr_lst, rw=[rw_files, rw_path], bl=bl,
                             uncorr_corrs=args.uncorr_corrs, uncorr_all=args.uncorr_all, return_gv=False, verbose=False)
    else:
        gv_data = ld.load_h5(fp.data_file, fp.corr_lst, bl=bl,
                             uncorr_corrs=args.uncorr_corrs, uncorr_all=args.uncorr_all)
        data_cfg = ld.load_h5(fp.data_file, fp.corr_lst, bl=bl,
                             uncorr_corrs=args.uncorr_corrs, uncorr_all=args.uncorr_all, return_gv=False, verbose=False)
    if args.states:
        states = args.states
    else:
        states = fp.fit_states

    x = copy.deepcopy(fp.x)
    y = {k: v[x[k]['t_range']]
         for (k, v) in gv_data.items() if k.split('_')[0] in states}
    for k in y:
        if 'exp_r' in x[k]['type']:
            sp = k.split('_')[-1]
            y[k] = y[k] / gv_data[x[k]['denom'][0]+'_'+sp][x[k]['t_range']]
            y[k] = y[k] / gv_data[x[k]['denom'][1]+'_'+sp][x[k]['t_range']]
    n_states = dict()
    for state in states:
        for k in x:
            if state in k:
                n_states[state] = x[k]['n_state']
    priors = dict()
    for k in fp.priors:
        for state in states:
            k_n = int(k.split('_')[-1].split(')')[0])
            if state == k.split('(')[-1].split('_')[0] and k_n < n_states[state]:
                priors[k] = gv.gvar(fp.priors[k].mean, fp.priors[k].sdev)

    if args.eff:
        
        plot.make_eff_plots(states=states,fp=fp,x_fit=None,priors=priors,gv_data=gv_data, fit=None,scale=args.scale,show_fit=False)
        plt.ion()

    # set up svdcut if added
    if args.svdcut is not None:
        svdcut = args.svdcut
        has_svd = True
    else:
        try:
            svdcut = fp.svdcut
            has_svd = True
        except:
            has_svd = False

    if args.stability:
        plot.make_stability_plot(states=states,x=x,fp=fp, priors=priors, gv_data=gv_data,scale=args.scale, stability=args.stability, svd_test=args.svd_test,
        svd_nbs=args.svd_nbs,data_cfg = data_cfg,n_states=n_states,es_stability=args.es_stability,save_figs=args.save_figs)

        # p = copy.deepcopy(fp.priors)

        # for state in args.stability:
        #     if 't_sweep' in fp.corr_lst[state]:
        #         tmin = fp.corr_lst[state]['t_sweep']
        #     else:
        #         tmin = range(2, x[state]['t_range'][-1])
        #     if 'n_sweep' in fp.corr_lst[state]:
        #         n_states = fp.corr_lst[state]['n_sweep']
        #     else:
        #         n_states = range(1, 6)
        #     tn_opt = (fp.corr_lst[state]['t_range'][0],
        #               fp.corr_lst[state]['n_state'])
        #     x_tmp = dict()
        #     for k in [kk for kk in x if kk.split('_')[0] == state]:
        #         x_tmp[k] = x[k].copy()

        #     fits = {}
        #     for ti in tmin:scale
        #     x_tmp[k]['t_range'] = np.arange(ti, x[k]['t_range'][-1]+1)

        #     y_tmp = {k: v[x_tmp[k]['t_range']]
        #                  for (k, v) in gv_data.items() if k in x_tmp}
        #     if args.svd_test:
        #         y_chop = dict()
        #         for d in y_tmp:
        #             if d in x_tmp:
        #                 y_chop[d] = data_cfg[d][:,x_tmp[d]['t_range']]
        #         s = gv.dataset.svd_diagnosis(y_chop, nbstrap=args.svd_nbs)
        #         svdcut = s.svdcut
        #         has_svd = True
        #     for k in x_tmp:
        #         if k.split('_')[0] not in states:
        #             y_tmp.pop(k)
        #     if ti == tmin[0]:
        #         print([k for k in y_tmp])
        #     for ns in n_states:
        #         xx = copy.deepcopy(x_tmp)
        #         for k in xx:
        #             ''' NOTE  - we are chaning n_s for pi, D and Dpi all together '''
        #             xx[k]['n_state'] = ns
        #         fit_funcs = cf.FitCorr()
        #         p_sweep = {}
        #         for k in p:
        #             if int(k.split('_')[-1].split(')')[0]) < ns:
        #                 p_sweep[k] = p[k]
        #         p0 = {k: v.mean for (k, v) in priors.items()}
        #         #print('t_min = %d  ns = %d' %(ti,ns))
        #         sys.stdout.write(
        #             'sweeping t_min = %d n_s = %d\r' % (ti, ns))
        #         sys.stdout.flush()
        #         if has_svd:
        #             f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
        #                                             prior=p_sweep, p0=p0,
        #                                             fcn=fit_funcs.fit_function, svdcut=svdcut)
        #         else:
        #             f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
        #                                             prior=p_sweep, p0=p0,
        #                                             fcn=fit_funcs.fit_function)
        #         fits[(ti, ns)] = f_tmp

        #     ylim = None
        #     if 'eff_ylim' in x_tmp[list(x_tmp.keys())[0]]:
        #         ylim = x_tmp[k]['eff_ylim']
        #     ylim = None
        #     plot.plot_stability(fits, tmin, n_states, tn_opt,
        #                         state, ylim=ylim, save=args.save_figs)
        #     if args.es_stability:
        #         for i_n in range(1, n_states[-1]):
        #             plot.plot_stability(fits, tmin, n_states, tn_opt, state,
        #                                 ylim=ylim, save=args.save_figs, n_plot=i_n, scale=args.scale)
        # print('')

    if args.fit:
        fit_funcs = cf.FitCorr()
        p0 = {k: v.mean for (k, v) in priors.items()}
        # only pass x for states in fit
        x_fit = dict()
        fit_lst = [k for k in x if k.split('_')[0] in states]
        for k in fit_lst:
            x_fit[k] = x[k]

        if args.svd_test:
            data_chop = dict()
            for d in y:
                if d in x_fit:
                    data_chop[d] = data_cfg[d][:,x_fit[d]['t_range']]
            svd_test = gv.dataset.svd_diagnosis(data_chop, nbstrap=args.svd_nbs)
            svdcut = svd_test.svdcut
            has_svd = True
            if args.svdcut is not None:
                print('    s.svdcut = %.2e' %svd_test.svdcut)
                print(' args.svdcut = %.2e' %args.svdcut)
                use_svd = input('   use specified svdcut instead of that from svd_diagnosis? [y/n]\n')
                if use_svd in ['y','Y','yes']:
                    svdcut = args.svdcut
        if has_svd:
            fit = lsqfit.nonlinear_fit(data=(x_fit, y), prior=priors, p0=p0, fcn=fit_funcs.fit_function,
                                       svdcut=svdcut)
        else:
            fit = lsqfit.nonlinear_fit(
                data=(x_fit, y), prior=priors, p0=p0, fcn=fit_funcs.fit_function)
        if args.verbose_fit:
            print(fit.format(maxline=True))
        else:
            print(fit)

        if args.gui:
            from lsqfitgui import run_server
            run_server(fit, name="c51 Two-Point Fitter")

        if args.eff:
            plot.make_eff_plots(states, fp, x_fit=x_fit, fit=fit,gv_data=gv_data, priors=priors, scale=args.scale,show_fit=True)

            # x_plot = copy.deepcopy(x_fit)
            # for k in x_plot:
            #     sp = k.split('_')[-1]
            #     ax = ax_meff[k.split('_')[0]]
            #     if 't0' in x_fit[k]:
            #         t0 = x_fit[k]['t0']
            #     else:
            #         t0 = 0
            #     x_plot[k]['t_range'] = np.arange(
            #         x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
            #     fit_funcs.corr_functions.eff_mass(
            #         x_plot[k], fit.p, ax, t0=t0, color=x_plot[k]['color'])
            #     x_plot[k]['t_range'] = np.arange(
            #         x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
            #     fit_funcs.corr_functions.eff_mass(
            #         x_plot[k], fit.p, ax, t0=t0, color='k', alpha=.1)
            #     if 'exp_r' in x_plot[k]['type']:
            #         ax = ax_r[k.split('_')[0]]
            #         if x_plot[k]['type'] in ['exp_r', 'exp_r_conspire']:
            #             x_plot[k]['t_range'] = np.arange(
            #                 x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
            #             x_plot[x_plot[k]['denom'][0]+'_'
            #                    + sp]['t_range'] = x_plot[k]['t_range']
            #             x_plot[x_plot[k]['denom'][1]+'_'
            #                    + sp]['t_range'] = x_plot[k]['t_range']
            #             d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
            #                    x_plot[x_plot[k]['denom'][1]+'_'+sp]]
            #             fit_funcs.corr_functions.eff_mass(
            #                 x_plot[k], fit.p, ax, color=x_plot[k]['color'], denom_x=d_x)
            #             x_plot[k]['t_range'] = np.arange(
            #                 x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
            #             x_plot[x_plot[k]['denom'][0]+'_'
            #                    + sp]['t_range'] = x_plot[k]['t_range']
            #             x_plot[x_plot[k]['denom'][1]+'_'
            #                    + sp]['t_range'] = x_plot[k]['t_range']
            #             d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
            #                    x_plot[x_plot[k]['denom'][1]+'_'+sp]]
            #             fit_funcs.corr_functions.eff_mass(
            #                 x_plot[k], fit.p, ax, color='k', alpha=.1, denom_x=d_x)
            #         else:
            #             x_plot[k]['t_range'] = np.arange(
            #                 x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
            #             fit_funcs.corr_functions.eff_mass(
            #                 x_plot[k], fit.p, ax, color=x_plot[k]['color'])
            #             x_plot[k]['t_range'] = np.arange(
            #                 x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
            #             fit_funcs.corr_functions.eff_mass(
            #                 x_plot[k], fit.p, ax, color='k', alpha=.1)

        if args.eff and args.scale and args.save_figs:
            plot.plot_eff_fit(states, fp, x_fit, scale,save_figs=True)
            # for k in ax_meff:
            #     s, units = float(args.scale[0]), args.scale[1]
            #     axr = ax_meff[k].twinx()
            #     print(k, ax_meff[k].get_ylim())
            #     print(ax_meff[k].get_yticks())
            #     axr.set_ylim(ax_meff[k].get_ylim()[0]*s,
            #                  ax_meff[k].get_ylim()[1]*s)
            #     axr.set_yticks([s*t for t in ax_meff[k].get_yticks()[:-1]])
            #     if units in ['GeV', 'gev']:
            #         axr.set_yticklabels(["%.2f" % t for t in axr.get_yticks()])
            #     else:
            #         axr.set_yticklabels(["%.0f" % t for t in axr.get_yticks()])
            #     axr.set_ylabel(r'$m_{\rm eff}(t) / {\rm %s}$' %
            #                    (units), fontsize=20)

        #if args.save_figs:
            # for k in states:
            #     n_s = str(fp.corr_lst[k]['n_state'])
            #     plt.figure('m_'+k)
            #     plt.savefig('figures/'+k+'_meff_ns'
            #                 + n_s+'.pdf', transparent=True)
            #     plt.figure('z_'+k)
            #     plt.savefig('figures/'+k+'_zeff_ns'
            #                 + n_s+'.pdf', transparent=True)
            #     if 'exp_r' in fp.corr_lst[k]['type']:
            #         plt.figure('r_'+k)
            #         plt.savefig('figures/'+k+'_ratio_meff_ns'
            #                     + n_s+'.pdf', transparent=True)

        if args.bs:
            # make sure results dir exists
            if not os.path.exists('bs_results'):
                os.makedirs('bs_results')
            if len(args.bs_results.split('/')) == 1:
                bs_file = 'bs_results/'+args.bs_results
            else:
                bs_file = args.bs_results
            # check if we already wrote this dataset
            have_bs = False
            if os.path.exists(bs_file):
                #with h5.open_file(bs_file,'r') as f5:
                with h5py.File(bs_file, 'r') as f5:
                    if args.bs_path in f5:
                        if len(f5[args.bs_path]) > 0 and not args.overwrite:
                            have_bs = True
                            print(
                                'you asked to write bs results to an existing dset and overwrite =', args.overwrite)
            if not have_bs:
                print('beginning Nbs=%d bootstrap fits' % args.Nbs)
                import bootstrap as bs

                # let us use the fit posterior to set the initial guess for bs loop
                p0_bs = dict()
                for k in fit.p:
                    p0_bs[k] = fit.p[k].mean

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

                # make BS data
                corr_bs = {}
                for k in data_cfg:
                    corr_bs[k] = bs.bs_corrs(data_cfg[k], Nbs=args.Nbs, seed=bs_seed, return_mbs=True)

                # make BS list for priors
                p_bs_mean = dict()
                for k in priors:
                    p_bs_mean[k] = bs.bs_prior(args.Nbs, mean=priors[k].mean,
                                            sdev=priors[k].sdev, seed=bs_seed+'_'+k)

                # set up posterior lists of bs results
                post_bs = dict()
                for k in fit.p:
                    post_bs[k] = []

                for bs in range(args.Nbs):
                    sys.stdout.write('%4d / %d\r' % (bs, args.Nbs))
                    sys.stdout.flush()

                    ''' all gvar's created in this switch are destroyed at restore_gvar [they are out of scope] '''
                    gv.switch_gvar()

                    bs_data = dict()
                    for k in corr_bs:
                        bs_data[k] = corr_bs[k][bs]
                    bs_gv = gv.dataset.avg_data(bs_data)
                    y_bs = {k: v[x_fit[k]['t_range']]
                            for (k, v) in bs_gv.items() if k in fit_lst}
                    p_bs = dict()
                    for k in p_bs_mean:
                        p_bs[k] = gv.gvar(p_bs_mean[k][bs], priors[k].sdev)
                    # do the fit
                    if has_svd:
                        fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                                      fcn=fit_funcs.fit_function, svdcut=svdcut)
                    else:
                        fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                                      fcn=fit_funcs.fit_function)

                    for r in post_bs:
                        post_bs[r].append(fit_bs.p[r].mean)

                    ''' end of gvar scope used for bootstrap '''
                    gv.restore_gvar()

                for r in post_bs:
                    post_bs[r] = np.array(post_bs[r])
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
                            f5.create_dataset(
                                args.bs_path+'/'+r, data=post_bs[r])

                print('DONE')

        if args.svd_test:
            fig = plt.figure('svd_diagnosis', figsize=(7, 4))
            svd_test.plot_ratio(show=True)
    if args.interact:
        import IPython; IPython.embed()

    plt.ioff()
    plt.show()

    if args.fit and args.gui:
        from lsqfitgui import run_server
        run_server(fit, name="c51 Two-Point Fitter")



if __name__ == "__main__":
    main()
