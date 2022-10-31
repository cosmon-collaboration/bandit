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

# c51_corr_analysis libs
import fitter.plotting as plot
import fitter.corr_functions as cf
import fitter.load_data as ld
import fitter.analysis as analysis

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
    parser.add_argument('--plot_name',   type=str, default='',
                        help=            'base name for figures [%(default)s]')
    parser.add_argument('--bs',          default=False, action='store_true',
                        help=            'run bootstrap fit? [%(default)s]')
    parser.add_argument('--Nbs',         type=int, default=2000,
                        help=            'specify the number of BS samples to compute [%(default)s]')
    parser.add_argument('--Mbs',         type=int, default=None,
                        help=            'number of random draws per bootstrap sample [%(default)s]')
    parser.add_argument('--bs_seed',     default=None,
                        help=            'set a string to seed the bootstrap - None will be random [%(default)s]')
    parser.add_argument('--bs0_restrict',default=True, action='store_false',
                        help=            'constrain ground state mean sampling with b0 posterior')
    parser.add_argument('--bs0_width',   type=float, default=5.0,
                        help=            'multiplication factor of posterior width for ground state prior mean sampling')
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

    # prepare x, y, priors
    x, y, priors = ld.prepare_xyp(states, fp, gv_data)

    if args.eff:
        eff_plots = plot.EffectivePlots()
        plt.ion()
        eff_plots.make_eff_plots(states, fp, priors, gv_data, scale=args.scale)

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

    # if user passes a base plot name
    if 'plot_name' in dir(fp):
        plot_name = fp.plot_name
    if args.plot_name:
        plot_name = args.plot_name

    # run a stability analysis
    if args.stability:
        analysis.run_stability(args, fp, x, y, gv_data, data_cfg, plot_name)

    if args.fit:
        fit_funcs = cf.FitCorr()
        # select data to be fit and p0 starting values
        x_fit, y_fit, p0 = fit_funcs.get_xyp0(priors, states, x, y)

        # are we using the SVD Diagnosis?
        if args.svd_test:
            has_svd = True
            svd_test, svdcut = ld.svd_diagnose(y, data_cfg, x_fit, args.svd_nbs,
                                               svdcut=args.svdcut)

        # run the fit
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

        # open a GUI if we want to interact with the priors and fit
        if args.gui:
            from lsqfitgui import run_server
            run_server(fit, name="c51 Two-Point Fitter")

        # plot the results on the eff mass data
        if args.eff:
            eff_plots.plot_eff_fit(x_fit, fit)

        if args.save_figs and args.eff:
            eff_plots.save_plots(name=plot_name)

        # Bootstrap
        if args.bs:
            bs_results, bs_fit_report = analysis.run_bootstrap(args, fit, fp, data_cfg, x_fit)

        if args.svd_test and args.eff:
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
