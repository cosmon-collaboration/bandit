import fitter.plotting as plot
import fitter.corr_functions as cf
import fitter.load_data as ld
import fitter.bootstrap as bootstrap
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
import tables as h5


# user libs
#sys.path.append('util')


def main():
    # do sys.argv routine here 
    parser = argparse.ArgumentParser(
        description='Perform analysis of two-point correlation function')
    parser.add_argument('fit_params',    help='input file to specify fit')
    parser.add_argument('--fit',         default=True, action='store_true',
                        help=            'do fit? [%(default)s]')
    parser.add_argument('--svdcut',      type=float, help='add svdcut to fit')
    parser.add_argument('--svd_test',    default=True, action='store_false',
                        help=            'perform gvar svd_diagnosis? [%(default)s]')
    parser.add_argument('--svd_nbs',     type=int, default=50, help='number of BS samples for estimating SVD cut [%(default)s]')
    parser.add_argument('--fold',        default=True, action='store_false',
                        help=            'fold data about T/2? [%(default)s]')
    parser.add_argument('-b', '--block', default=1, type=int,
                        help=            'specify bin/blocking size in terms of saved configs')
    parser.add_argument('--eff',         default=True, action='store_true',
                        help=            'plot effective mass and z_eff data? [%(default)s]')
    parser.add_argument('--mres',         default=False, action='store_true',
                        help=            'plot mres avg data with fit? [%(default)s]')
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
    parser.add_argument('--bs_write',    default=True, action='store_false',
                        help=            'write bs results to file? [%(default)s]')
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

    # pulling in prepared x,y data, n_states from input file, and necessary priors to run fit 
    x,y,n_states,priors= ld.make_fit_params(fp=fp,states=states,gv_data=gv_data)
    
    if args.eff:
        #plt.ion()
        effective = plot.eff_plots()
        effective.make_eff_plots(states=states,fp=fp,x_fit=None,priors=priors,
        gv_data=gv_data,fit=None, scale=args.scale,show_fit=False,save_figs=args.save_figs,x=x)
    # if args.mres:
    #     plot.plot_mres(ax, dsets, key, svdcut=args.svdcut, stability=args.stability)

    #set up svdcut if added
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
        plot.make_stability_plot(states=args.states,x=x,fp=fp,gv_data=gv_data, stability=args.stability,priors=priors, 
        scale = args.scale, svd_test=args.svd_test, data_cfg = data_cfg,n_states=n_states, 
        svd_nbs=args.svd_nbs, es_stability=args.es_stability,save_figs=args.save_figs)

    if args.fit:
        fit_funcs = cf.FitCorr()
        p0, x_fit, y_fit = fit_funcs.get_fit(priors=priors, states=states,x=x,y=y)
        if args.svd_test:
            data_chop = dict()
            for d in y:
                if d in x_fit:
                    if d in x_fit and 'mres' not in d:
                        data_chop[d] = data_cfg[d][:,x_fit[d]['t_range']]
                    if 'mres' in d and len(d.split('_')) > 1:
                        data_chop[d] = data_cfg[d][:,x_fit[d.split('_')[0]]['t_range']]
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
            fit = lsqfit.nonlinear_fit(data=(x_fit, y_fit), prior=priors, p0=p0, fcn=fit_funcs.fit_function,
                                       svdcut=svdcut)
        else:
            fit = lsqfit.nonlinear_fit(
                data=(x_fit, y_fit), prior=priors, p0=p0, fcn=fit_funcs.fit_function)
        if args.verbose_fit:
            print(fit.format(maxline=True))
        else:
            print(fit)

        if args.gui:
            from lsqfitgui import run_server
            run_server(fit, name="c51 Two-Point Fitter")

        if args.eff:
            #x_plot = copy.deepcopy(x_fit)
            effective.make_eff_plots(states=states, fp=fp, x_fit=x_fit,fit=fit,gv_data=gv_data, priors=priors, 
                                scale=args.scale,show_fit=True,save_figs=args.save_figs,x=x)
        # run bootstrapping routine: 
        # - ensure bs_dir exits
        # - run bs loop with fit posterior as initial guess 
        # - set seed if not defined 
        # - make BS data 
        # - make BS list for priors
        # - set up posterior lists of bs results 
        # - write out bs results 
        if args.bs:
            bootstrap.run_bs(bs_write=args.bs_write, bs_results=args.bs_results, 
                                bs_path=args.bs_path,overwrite=args.overwrite,Nbs=args.Nbs, fit=fit,
                                bs_seed=args.bs_seed, verbose=args.verbose, data_cfg=data_cfg, priors=priors,
                                states=states,has_svd=has_svd, fp =fp,x_fit=x_fit,svdcut=svdcut)

        if args.svd_test:
            fig = plt.figure('svd_diagnosis', figsize=(7, 4))
            svd_test.plot_ratio(show=True)
    if args.interact:
        import IPython; IPython.embed()

    plt.ioff()
    plt.show()

    # if args.fit and args.gui:
    #     from lsqfitgui import run_server
    #     run_server(fit, name="c51 Two-Point Fitter")



if __name__ == "__main__":
    main()
