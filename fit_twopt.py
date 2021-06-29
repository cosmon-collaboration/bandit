import os, sys, random
import matplotlib.pyplot as plt
import argparse
import numpy as np
np.seterr(invalid='ignore')
import copy
#import tables as h5
import h5py

import gvar as gv
import lsqfit

# user libs
sys.path.append('util')
import load_data as ld
import corr_functions as cf
import plotting as plot

def main(args):
    import fit_params as fp
    gv_data = ld.load_h5(fp.data_file, fp.corr_lst)


    if args.states:
        states = args.states
    else:
        states = fp.fit_states

    x = copy.deepcopy(fp.x)
    y = {k:v[x[k]['t_range']] for (k,v) in gv_data.items() if k.split('_')[0] in states}
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
        plt.ion()
        ax_meff = {}
        ax_zeff = {}
        for k in states:
            clrs = fp.corr_lst[k]['colors']
            # m_eff
            fig   = plt.figure('m_'+k, figsize=(7,4))
            ax_meff[k]= plt.axes([0.15,0.15,0.84,0.84])
            p = priors[k+'_E_0']
            ax_meff[k].axhspan(p.mean-p.sdev, p.mean+p.sdev, color='k',alpha=.2)
            plot.plot_eff(ax_meff[k], gv_data, k, mtype=fp.corr_lst[k]['type'], colors=clrs)
            ax_meff[k].set_xlim(fp.corr_lst[k]['xlim'])
            ax_meff[k].set_ylim(fp.corr_lst[k]['ylim'])
            ax_meff[k].set_xlabel(r'$t/a$', fontsize=20)
            ax_meff[k].set_ylabel(r'$m_{\rm eff}^{\rm %s}(t)$' %state, fontsize=20)
            ax_meff[k].legend(fontsize=20)

            # z_eff
            fig   = plt.figure('z_'+k, figsize=(7,4))
            ax_zeff[k]= plt.axes([0.15,0.15,0.84,0.82])
            snksrc = {'snks':fp.corr_lst[k]['snks'], 'srcs':fp.corr_lst[k]['srcs']}
            mtype  = fp.corr_lst[k]['type']
            ztype  = fp.corr_lst[k]['ztype']
            # this zeff prior is not generic yet
            for z in fp.corr_lst[k]['snks']:
                p = priors[k+'_z'+z+'_0']
                ax_zeff[k].axhspan(p.mean-p.sdev, p.mean+p.sdev, color='k',alpha=.2)
            plot.plot_zeff(ax_zeff[k], gv_data, k, ztype=ztype, mtype=mtype, snksrc=snksrc, colors=clrs)
            ax_zeff[k].set_xlim(fp.corr_lst[k]['xlim'])
            ax_zeff[k].set_ylim(fp.corr_lst[k]['z_ylim'])
            ax_zeff[k].set_xlabel(r'$t/a$', fontsize=20)
            ax_zeff[k].set_ylabel(r'$z_{\rm eff}^{\rm %s}(t)$' %state, fontsize=20)
            ax_zeff[k].legend(fontsize=20, loc=1)

    if args.sweep:
        fit_funcs = cf.FitCorr()
        p  = copy.deepcopy(fp.priors)
        p0 = {k:v.mean for (k,v) in priors.items()}

        for state in args.sweep:
            if 't_sweep' in fp.corr_lst[state]:
                tmin = fp.corr_lst[state]['t_sweep']
            else:
                tmin = range(2,x[state]['t_range'][-1])
            if 'n_sweep' in fp.corr_lst[state]:
                n_states = fp.corr_lst[state]['n_sweep']
            else:
                n_states = range(1,6)
            tn_opt = (fp.corr_lst[state]['t_range'][0], fp.corr_lst[state]['n_state'])
            x_tmp = dict()
            for k in [kk for kk in x if kk.split('_')[0] == state]:
                x_tmp[k] = x[k].copy()

            fits = {}
            for ti in tmin:
                for k in x_tmp:
                    x_tmp[k]['t_range'] = np.arange(ti,x[k]['t_range'][-1]+1)
                xx = copy.deepcopy(x_tmp)
                y_tmp = {k:v[xx[k]['t_range']] for (k,v) in gv_data.items() if k in x_tmp}
                for k in x_tmp:
                    if k.split('_')[0] not in states:
                        y_tmp.pop(k)
                if ti == tmin[0]:
                    print([k for k in y_tmp])
                for ns in n_states:
                    sys.stdout.write('sweeping t_min = %d n_s = %d\r' %(ti,ns))
                    sys.stdout.flush()
                    for k in xx:
                        ''' NOTE  - we are chaning n_s for pi, D and Dpi all together '''
                        xx[k]['n_state'] = ns
                    f_tmp = lsqfit.nonlinear_fit(data=(xx,y_tmp), prior=p, p0=p0, fcn=fit_funcs.fit_function)
                    fits[(ti,ns)] = f_tmp
            ylim=None
            if 'eff_ylim' in x_tmp[k]:
                ylim = x_tmp[k]['eff_ylim']
            plot.plot_stability(fits, tmin, n_states, tn_opt, state, ylim=ylim, save=args.save_figs)
        print('')

    if args.fit:
        fit_funcs = cf.FitCorr()
        p0 = {k:v.mean for (k,v) in priors.items()}
        # only pass x for states in fit
        x_fit = dict()
        fit_lst = [k for k in x if k.split('_')[0] in states]
        for k in fit_lst:
            x_fit[k] = x[k]

        fit = lsqfit.nonlinear_fit(data=(x_fit,y), prior=priors, p0=p0, fcn=fit_funcs.fit_function)
        if args.verbose_fit:
            print(fit.format(maxline=True))
        else:
            print(fit)

        if args.eff:
            x_plot = copy.deepcopy(x_fit)
            for k in x_plot:
                ax = ax_meff[k.split('_')[0]]
                x_plot[k]['t_range'] = np.arange(x[k]['t_range'][0],x[k]['t_range'][-1]+.1,.1)
                fit_funcs.corr_functions.eff_mass(x_plot[k], fit.p, ax, color=x_plot[k]['color'])
                x_plot[k]['t_range'] = np.arange(x[k]['t_range'][-1]+.5,x[k]['t_range'][-1]+20.1,.1)
                fit_funcs.corr_functions.eff_mass(x_plot[k], fit.p, ax, color='k', alpha=.1)
        if args.save_figs:
            for k in states:
                plt.figure('m_'+k)
                plt.savefig('figures/'+k+'_meff.pdf', transparent=True)
                plt.figure('z_'+k)
                plt.savefig('figures/'+k+'_zeff.pdf', transparent=True)

        if args.bs:
            # make sure results dir exists
            if not os.path.exists('bs_results'):
                os.makedirs('bs_results')
            if len(args.bs_results.split('/')) == 1:
                bs_file='bs_results/'+args.bs_results
            else:
                bs_file=args.bs_results
            # check if we already wrote this dataset
            have_bs = False
            if os.path.exists(bs_file):
                #with h5.open_file(bs_file,'r') as f5:
                with h5py.File(bs_file,'r') as f5:
                    if args.bs_path in f5:
                        if len(f5[args.bs_path]) > 0 and not args.overwrite:
                            have_bs = True
                            print('you asked to write bs results to an existing dset and overwrite =',args.overwrite)
            if not have_bs:
                print('beginning Nbs=%d bootstrap fits' %args.Nbs)
                # let us use the fit posterior to set the initial guess for bs loop
                p0_bs = dict()
                for k in fit.p:
                    p0_bs[k] = fit.p[k].mean

                # load the data
                corr_data = ld.load_h5(fp.data_file, fp.corr_lst, return_gv=False)
                Ncfg = corr_data[list(corr_data.keys())[0]].shape[0]

                # seed the random number generator
                if args.bs_seed == 'None':
                    random.seed(None)
                else:
                    try:
                        bs_seed = fp.bs_seed
                    except:
                        bs_seed = args.bs_seed
                    random.seed(args.bs_seed)
                seed_int = random.randint(1,1e6)
                np.random.seed(seed_int)
                # make BS list and random prior central values
                bs_lst = np.random.randint(Ncfg, size=[args.Nbs, Ncfg])
                p_bs_mean = dict()
                for k in priors:
                    p_bs_mean[k] = np.random.normal(loc=priors[k].mean, scale=priors[k].sdev, size=args.Nbs)

                gs_bs = dict()
                for k in fit.p:
                    if k.split('_')[-1] == '0':
                        gs_bs[k] = []

                for bs in range(args.Nbs):
                    sys.stdout.write('%4d / %d\r' %(bs,args.Nbs))
                    sys.stdout.flush()
                    corr_bs = {}
                    for k in corr_data:
                        corr_bs[k] = corr_data[k][bs_lst[bs]]
                    bs_gv = gv.dataset.avg_data(corr_bs)
                    y_bs = {k:v[x_fit[k]['t_range']] for (k,v) in bs_gv.items() if k in states}
                    p_bs = dict()
                    for k in p_bs_mean:
                        p_bs[k] = gv.gvar(p_bs_mean[k][bs], priors[k].sdev)
                    # do the fit
                    fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs, fcn=fit_funcs.fit_function)
                    for r in gs_bs:
                        if r in fit_bs.p:
                            gs_bs[r].append(fit_bs.p[r].mean)
                for r in gs_bs:
                    gs_bs[r] = np.array(gs_bs[r])
                # write the results
                with h5py.File(bs_file,'a') as f5:
                    try:
                        f5.create_group(args.bs_path)
                    except Exception as e:
                        print(e)
                    for r in gs_bs:
                        if len(gs_bs[r]) > 0:
                            if r in f5[args.bs_path]:
                                del f5[args.bs_path+'/'+r]
                            f5.create_dataset(args.bs_path+'/'+r, data=gs_bs[r])

                print('DONE')
    if args.interact:
        import IPython; IPython.embed()


    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform analysis of two-point correlation function')
    parser.add_argument('--fit',         default=False, action='store_true',
                        help=            'do fit? [%(default)s]')
    parser.add_argument('--fold',        default=True, action='store_false',
                        help=            'fold data about T/2? [%(default)s]')
    parser.add_argument('--eff',         default=False, action='store_true',
                        help=            'plot effective mass and z_eff data? [%(default)s]')
    parser.add_argument('--sweep',       nargs='+',
                        help=            'perform sweep of t_min and n_state for specified states')
    parser.add_argument('--states',      nargs='+', help='specify states to fit?')
    parser.add_argument('--verbose_fit', default=False, action='store_true',
                        help=            'print y vs f(x,p) also? [%(default)s]')
    parser.add_argument('--save_figs',   default=False, action='store_true',
                        help=            'save figs? [%(default)s]')
    parser.add_argument('--bs',          default=False, action='store_true',
                        help=            'run bootstrap fit? [%(default)s]')
    parser.add_argument('--Nbs',         type=int, default=2000,
                        help=            'specify the number of BS samples to compute [%(default)s]')
    parser.add_argument('--bs_seed',     default='None',
                        help=            'set a string to seed the bootstrap - None will be random [%(default)s]')
    parser.add_argument('--bs_results',  default='bs_results/spectrum_bs.h5',
                        help=            'set file to write bootstrap results [%(default)s]')
    parser.add_argument('--overwrite',   default=False, action='store_true',
                        help=            'overwrite existing bootstrap results? [%(default)s]')
    parser.add_argument('--bs_path',     default='spec',
                        help=            'specify path in h5 file for bs results')
    parser.add_argument('--interact',    default=False, action='store_true',
                        help=            'open IPython instance after to interact with results? [%(default)s]')

    args = parser.parse_args()
    if args.save_figs and not os.path.exists('figures'):
        os.makedirs('figures')
    print(args)
    main(args)
