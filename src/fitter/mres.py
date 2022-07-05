def plot_mres(ax, dsets, key, svdcut,stability,mtype='exp', tau=1, colors=None, offset=0,n_plot=0):
    lst = [k for k in dsets if key in k]
    mres = lst[0].split('_')[0]
    data = dsets[mres+'_MP'] / dsets[mres+'_PP']
    Nt = dsets[mres+'_MP'].shape[0]
    mres_avg =  np.mean(data[10:Nt//2+1]) 
    lbl  = mres
    ax.fill_between(np.arange(10,Nt/2+1,.1), mres_avg.mean-mres_avg.sdev, mres_avg.mean+mres_avg.sdev, color='r',alpha=.3, label='avg')
    # do fit 
    p = copy.deepcopy(fp.priors)
    x_fit = np.arange(data.shape[0])
    y_fit = mres_avg[x_fit]
    # svd diagnosis 
    d_fit = {}
    d_fit['MP'] = dsets[mres+'_MP'][x_fit]
    d_fit['PP'] = dsets[mres+'_MP'][x_fit]
    s = gv.dataset.svd_diagnosis(d_fit)
    svdcut = s.svdcut
    print('---------------------------------------------------------------------')
    print(ens)
    print('---------------------------------------------------------------------')
    print('svd_diagnosis - suggested svd cut = %s\n' %s.svdcut)
    if args.svdcut:
        print('over-riding svd_diagnosis suggested cut with user specified SVDCUT = %f' %args.svdcut)
        svdcut = args.svdcut
    fit = lsqfit.nonlinear_fit(data=(x_fit, y_fit), prior=p, fcn=fit_funcs.mres_func, svdcut=svdcut)
    print(fit)
    # plot fit
    ax.fill_between(x_fit, fit.p['mres'].mean-fit.p['mres'].sdev, fit.p['mres'].mean+fit.p['mres'].sdev, color='b',alpha=.3,label='fit')

    # data
    y  = [k.mean for k in mres_avg]
    dy = [k.sdev for k in mres_avg]
    ax.errorbar(x,y,yerr=dy,linestyle='None',marker='s',mfc='None',color='k', alpha=.5, label=ens)

    ax.set_ylabel(r'$m_{\rm res}$',fontsize=20)
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylim(y_scale[ens])
    ax.set_xlim(0,Nt/2+0.5)

    ax.legend(loc=1,fontsize=16)
    nn = str(n_plot)
    if stability:
        for state in states:

            plt.figure(state+'_E_'+nn+'_stability', figsize=(7, 4))
            ax = plt.axes([0.12,.12,.85,.85])
            for t in range(0,Nt//2):
                # correlated average
                if t == 0:
                    lbl = 'avg'
                    lbl_f = 'fit'
                else:
                    lbl = ''
                    lbl_f = ''
                ax.errorbar(t, mres_avg.mean, yerr=mres_avg.sdev, marker='s', mfc='None', color='r',alpha=.3, linestyle='None', label=lbl)

                # fit
                
                x_fit = np.arange(t,Nt//2+1)
                y_fit = mres_avg[x_fit]
                fit = lsqfit.nonlinear_fit(data=(x_fit, y_fit), prior=p, fcn= fit_funcs.mres_func, svdcut=svdcut)

                #import IPython; IPython.embed()
                # plot fit
                ax.errorbar(t, fit.p['mres'].mean, yerr=fit.p['mres'].sdev, marker='o', mfc='None', linestyle='None', color='b',alpha=.3,label=lbl_f)

        ax.legend(loc=1,fontsize=16)

        ax.set_ylabel(r'$\tilde{m}_{\rm res}$',fontsize=20)
        ax.set_xlabel(r'$t_{\rm min}$', fontsize=20)
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0,Nt/2+0.5)

        if savefigs:
            for k in states:
                n_s = str(fp.corr_lst[k]['n_state'])
                plt.figure('mres_'+k)
                plt.savefig('figures/'+k+'_meff_ns'
                            + n_s+'.pdf', transparent=True)
            if not os.path.exists('figures'):
                os.makedirs('figures')
            