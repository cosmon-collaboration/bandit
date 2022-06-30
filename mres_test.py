import os
import tables as h5
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import lsqfit

import argparse

def main():
    parser = argparse.ArgumentParser(description='m_res analysis')
    parser.add_argument('--svdcut', type=float, help='set svdcut [%(default)s]')
    parser.add_argument('--savefigs', default=False, action='store_true', help='save figures? [%(default)s]')
    args = parser.parse_args()
    svdcut=args.svdcut

    # fold data functions
    def fold_corr(corr, time_axis=1):
        ''' assumes time index is second of array
        '''
        if len(corr.shape) > 1:
            cr = np.roll(corr[:, ::-1], 1, axis=time_axis)
        else:
            cr = np.roll(corr[::-1], 1)

        return 0.5 * (cr + corr)

    # load data
    ensembles = ['a12m310', 'a06m310']
    d_name = {
        'a12m310_MP':'mp_ml0p0126',  'a12m310_PP':'pp_ml0p0126',
        'a06m310_MP':'MP_ml0p00614', 'a06m310_PP':'PP_ml0p00614',
        }
    data = dict()
    with h5.open_file('mres_test.h5','r') as f5:
        for ens in ensembles:
            for dtype in ['MP','PP']:
                corr = f5.get_node('/'+ens+'/'+d_name[ens+'_'+dtype]).read()
                data[ens+'_'+dtype] = fold_corr(corr)

    data_gv = gv.dataset.avg_data(data)


    y_scale = {'a12m310':(0.000725,0.00085), 'a06m310':(1.75e-5, 3.25e-5)}

    def mres_func(x,p):
        return p['mres']*np.ones(x.shape[0])

    mres_p = {'a12m310':gv.gvar(.0008, .0008), 'a06m310': gv.gvar(2.5e-5, 2.5e-5)}

    plt.ion()
    for ens in ensembles:
        plt.figure(ens+'_eff')
        ax = plt.axes([0.12,.12,.85,.85])
        Nt = data_gv[ens+'_MP'].shape[0]
        x  = np.arange(Nt)
        # correlated average
        mres = data_gv[ens+'_MP'] / data_gv[ens+'_PP']
        mres_avg = np.mean(mres[10:Nt//2+1])
        ax.fill_between(np.arange(10,Nt/2+1,.1), mres_avg.mean-mres_avg.sdev, mres_avg.mean+mres_avg.sdev, color='r',alpha=.3, label='avg')

        # fit
        p = dict()
        p['mres'] = mres_p[ens]
        x_fit = np.arange(10,Nt//2+1)
        y_fit = mres[x_fit]
        # svd diagnosis
        d_fit = {}
        d_fit['MP'] = data[ens+'_MP'][x_fit]
        d_fit['PP'] = data[ens+'_PP'][x_fit]
        s = gv.dataset.svd_diagnosis(d_fit)
        svdcut = s.svdcut
        print('---------------------------------------------------------------------')
        print(ens)
        print('---------------------------------------------------------------------')
        print('svd_diagnosis - suggested svd cut = %s\n' %s.svdcut)
        if args.svdcut:
            print('over-riding svd_diagnosis suggested cut with user specified SVDCUT = %f' %args.svdcut)
            svdcut = args.svdcut

        fit = lsqfit.nonlinear_fit(data=(x_fit, y_fit), prior=p, fcn=mres_func, svdcut=svdcut)
        print(fit)

        # plot fit
        ax.fill_between(x_fit, fit.p['mres'].mean-fit.p['mres'].sdev, fit.p['mres'].mean+fit.p['mres'].sdev, color='b',alpha=.3,label='fit')

        # data
        y  = [k.mean for k in mres]
        dy = [k.sdev for k in mres]
        ax.errorbar(x,y,yerr=dy,linestyle='None',marker='s',mfc='None',color='k', alpha=.5, label=ens)

        ax.set_ylabel(r'$m_{\rm res}$',fontsize=20)
        ax.set_xlabel(r'$t$', fontsize=20)
        ax.set_ylim(y_scale[ens])
        ax.set_xlim(0,Nt/2+0.5)

        ax.legend(loc=1,fontsize=16)


        # stability plots
        plt.figure(ens+'_stability')
        ax = plt.axes([0.12,.12,.85,.85])
        for t in range(0,Nt//2):
            # correlated average
            mres_avg = np.mean(mres[t:Nt//2+1])
            if t == 0:
                lbl = 'avg'
                lbl_f = 'fit'
            else:
                lbl = ''
                lbl_f = ''
            ax.errorbar(t, mres_avg.mean, yerr=mres_avg.sdev, marker='s', mfc='None', color='r',alpha=.3, linestyle='None', label=lbl)

            # fit
            p = dict()
            p['mres'] = mres_p[ens]
            x_fit = np.arange(t,Nt//2+1)
            y_fit = mres[x_fit]
            fit = lsqfit.nonlinear_fit(data=(x_fit, y_fit), prior=p, fcn=mres_func, svdcut=svdcut)

            #import IPython; IPython.embed()
            # plot fit
            ax.errorbar(t, fit.p['mres'].mean, yerr=fit.p['mres'].sdev, marker='o', mfc='None', linestyle='None', color='b',alpha=.3,label=lbl_f)

        ax.legend(loc=1,fontsize=16)

        ax.set_ylabel(r'$\tilde{m}_{\rm res}$',fontsize=20)
        ax.set_xlabel(r'$t_{\rm min}$', fontsize=20)
        ax.set_ylim(y_scale[ens])
        ax.set_xlim(0,Nt/2+0.5)

        if args.savefigs:
            if not os.path.exists('figures'):
                os.makedirs('figures')
            for ftype in ['eff','stability']:
                plt.figure(ens+'_'+ftype)
                plt.savefig('figures/'+ens+'_'+ftype+'.pdf',transparent=True)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
