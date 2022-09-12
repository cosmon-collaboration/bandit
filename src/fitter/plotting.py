import matplotlib.pyplot as plt
import numpy as np
import os
import copy
# our libs
import fitter.corr_functions as cf


figsize=(6.5, 6.5/1.618034333)

class EffectivePlots():

    def __init__(self):
        self.ax = {}

    def make_eff_plots(self,
        states,          # what states to make plot for
        params,          # user specified plotting parameters
        priors,          # user specified priors
        gv_data,         # dictionary containing data in gvar format
        scale=[] # show plots live?
        ):
        ''' Make dictionary of effective mass, and effective overlap factor plots

            states: list of states to make effective plots for
            params:
            priors:
            gv_data:
            scale:
        '''
        for k in states:
            # make this optional
            clrs = params.corr_lst[k]['colors']

            # does the data start from t=0?
            if 't0' in params.corr_lst[k]:
                t0 = params.corr_lst[k]['t0']
            else:
                t0 = 0

            # create effective mass plot
            fig = plt.figure('m_'+k, figsize=figsize)
            # save space on right if we add physical scale
            if scale:
                self.ax['m_'+k] = plt.axes([0.15, 0.15, 0.74, 0.84])
            else:
                self.ax['m_'+k] = plt.axes([0.15, 0.15, 0.84, 0.84])

            print(type(self.ax['m_'+k]))

            # plot prior on eff plot
            if params.corr_lst[k]['type'] == 'mres':
                p = priors[k]
            elif params.corr_lst[k]['type'] in ['exp_r', 'exp_r_conspire']:
                d1, d2 = params.corr_lst[k]['denom']
                p = priors[d1+'_E_0'] + priors[d2+'_E_0'] + priors[k+'_dE_0_0']
            else:
                p = priors[k+'_E_0']
            self.ax['m_'+k].axhspan(p.mean -p.sdev, p.mean +p.sdev, color='k', alpha=.2)

            # plot data
            if 'mres' not in k:
                plot_eff(self.ax['m_'+k], gv_data, k, mtype=params.corr_lst[k]['type'],
                        colors=clrs, offset=t0)
            else:
                plot_mres(self.ax['m_'+k], gv_data, k, mtype=params.corr_lst[k]['type'],
                        colors=clrs, offset=t0)
            # set ranges and labels
            self.ax['m_'+k].set_xlim(params.corr_lst[k]['xlim'])
            self.ax['m_'+k].set_ylim(params.corr_lst[k]['ylim'])
            self.ax['m_'+k].set_xlabel(r'$t/a$', fontsize=20)
            self.ax['m_'+k].set_ylabel(r'$m_{\rm eff}^{\rm %s}(t)$' % k, fontsize=20)
            self.ax['m_'+k].legend(fontsize=20)

            if scale:
                s, units = float(scale[0]), scale[1]
                axr = self.ax['m_'+k].twinx()
                axr.set_ylim(self.ax['m_'+k].get_ylim()[0]*s,
                             self.ax['m_'+k].get_ylim()[1]*s)
                axr.set_yticks([s*t for t in self.ax['m_'+k].get_yticks()[:-1]])
                if units in ['GeV', 'gev']:
                    axr.set_yticklabels(["%.2f" % t for t in axr.get_yticks()])
                else:
                    axr.set_yticklabels(["%.0f" % t for t in axr.get_yticks()])
                axr.set_ylabel(r'$m_{\rm eff}(t) / {\rm %s}$' %
                               (units), fontsize=20)

            # make effective z plots
            if 'mres' not in k:
                num_snks = len(params.corr_lst[k]['snks'])
                fig, axes = plt.subplots(nrows=num_snks, sharex=True, gridspec_kw={'hspace':0.1})

                snksrc = {'snks': params.corr_lst[k]['snks'],
                          'srcs': params.corr_lst[k]['srcs']}
                mtype = params.corr_lst[k]['type']
                ztype = params.corr_lst[k]['ztype']
                plot_zeff(axes, gv_data, k, ztype=ztype, mtype=mtype,
                        snksrc=snksrc, colors=clrs)

                axes[-1].set_xlim(params.corr_lst[k]['xlim'])
                axes[-1].set_xlabel(r'$t/a$', fontsize=20)
                for j, ax in enumerate(axes):
                    ax.legend(fontsize=20, loc=1)
                    ax.set_ylabel(r'$z_{\rm eff}^{\rm %s}(t)$' % k, fontsize=20)

                    # NOTE: there should be a different ylim for each snk
                    ax.set_ylim(params.corr_lst[k]['z_ylim'])
                    self.ax['z_%s_%s'%(k, str(j))] = ax


    def plot_eff_fit(self,
        x_fit,
        fit, ):
        '''
            x_fit: parameters for fit
            fit  : result of lsqfit.nonlinear_fit
        '''
        fit_funcs = cf.FitCorr()

        x_plot = copy.deepcopy(x_fit)
        for k_sp in x_plot:
            k,sp = k_sp.split('_')
            ax = self.ax['m_'+k]

            if 't0' in x_fit[k_sp]:
                t0 = x_fit[k_sp]['t0']
            else:
                t0 = 0

            # plot in fit region
            xi = x_fit[k_sp]['t_range'][0]
            xf = x_fit[k_sp]['t_range'][-1]
            x_plot[k_sp]['t_range'] = np.arange(xi, xf+.1, .1)
            fit_funcs.corr_functions.eff_mass(
                x_plot[k_sp], fit.p, ax, t0=t0, color=x_plot[k_sp]['color'])
            # extrapolate to large t
            x_plot[k_sp]['t_range'] = np.arange(xf+.5, xf+50.1, .1)
            fit_funcs.corr_functions.eff_mass(
                x_plot[k_sp], fit.p, ax, t0=t0, color='k', alpha=.2)

            if 'exp_r' in x_plot[k_sp]['type']:
                sys.exit('please enable eff plotting support for ratio fits')
                """
                ax = ax_r[k.split('_')[0]]
                if x_plot[k]['type'] in ['exp_r', 'exp_r_conspire']:
                    x_plot[k]['t_range'] = np.arange(
                        x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
                    x_plot[x_plot[k]['denom'][0]+'_'+ sp]['t_range'] = x_plot[k]['t_range']
                    x_plot[x_plot[k]['denom'][1]+'_'+ sp]['t_range'] = x_plot[k]['t_range']
                    d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
                           x_plot[x_plot[k]['denom'][1]+'_'+sp]]
                    fit_funcs.corr_functions.eff_mass(
                        x_plot[k], fit.p, ax, color=x_plot[k]['color'], denom_x=d_x)
                    x_plot[k]['t_range'] = np.arange(
                        x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
                    x_plot[x_plot[k]['denom'][0]+'_'
                           + sp]['t_range'] = x_plot[k]['t_range']
                    x_plot[x_plot[k]['denom'][1]+'_'
                           + sp]['t_range'] = x_plot[k]['t_range']
                    d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
                           x_plot[x_plot[k]['denom'][1]+'_'+sp]]
                    fit_funcs.corr_functions.eff_mass(
                        x_plot[k], fit.p, ax, color='k', alpha=.1, denom_x=d_x)
                else:
                    x_plot[k]['t_range'] = np.arange(
                        x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
                    fit_funcs.corr_functions.eff_mass(
                        x_plot[k], fit.p, ax, color=x_plot[k]['color'])
                    x_plot[k]['t_range'] = np.arange(
                        x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
                    fit_funcs.corr_functions.eff_mass(
                        x_plot[k], fit.p, ax, color='k', alpha=.1)
                """

    def save_plots(self):
        if not os.path.exists('figures'):
            os.makedirs('figures')
        for k in self.ax:
            plt.figure(k)
            plt.savefig('figures/'+k+'.pdf',transparent=True)

def effective_mass(gvdata, mtype='exp', tau=1):
    ''' Create effective mass data from gvar of the correlation function
        versus time.
        gvdata = array of gvar data of correlation function
        mtype  = type of effective mass: exp, cosh, cosh_costant, ...
        tau    = shift variable for making effective mass
    '''
    if 'exp' in mtype:
        meff = 1./tau * np.log(gvdata / np.roll(gvdata, -tau))
    elif mtype == 'cosh':
        meff = 1./tau * \
            np.arccosh((np.roll(gvdata, -tau)+np.roll(gvdata, tau))/2/gvdata)
    # chop off last time slice for wrap-around effects
    return meff[:-1]


def plot_eff(ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0, denom_key=None):
    lst = [k for k in dsets if key in k]
    for k in lst:
        data = dsets[k]
        if denom_key:
            for d in denom_key:
                data = data / dsets[k.replace(key, d)]
        lbl = k.split('_')[1]
        eff = effective_mass(data, mtype=mtype, tau=tau)
        x = np.arange(eff.shape[0]) + offset
        m = [k.mean for k in eff]
        dm = [k.sdev for k in eff]
        if colors is not None:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        color=colors[lbl], mfc='None', label=lbl)
        else:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        mfc='None', label=label)

def plot_mres(ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0, denom_key=None):
    lst = [k for k in dsets if key in k]
    mres = lst[0].split('_')[0]
    data = dsets[mres+'_MP'] / dsets[mres+'_PP']
    lbl  = mres
    x  = np.arange(data.shape[0]) + offset
    m  = [k.mean for k in data]
    dm = [k.sdev for k in data]
    if colors is not None:
        ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                    color=colors, mfc='None', label=lbl)
    else:
        ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                    mfc='None', label=label)


def plot_zeff(axes, dsets, key, ztype='A_snk,src', snksrc=None, mtype='exp', tau=1, colors=None):
    lst = [k for k in dsets if key in k]
    if ztype == 'A_snk,src':
        for j, k in enumerate(lst):
            lbl = k.split('_')[-1]
            eff = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t = np.arange(eff.shape[0])
            if 'exp' in mtype:
                zeff = np.exp(eff * t) * dsets[k][:-1]
            elif mtype == 'cosh':
                zeff = dsets[k][:-1] / \
                    (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            z = [k.mean for k in zeff]
            dz = [k.sdev for k in zeff]
            if colors is not None:
                axes.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            color=colors[lbl], mfc='None', label=lbl)
            else:
                axes.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            mfc='None', label=lbl)
    elif ztype == 'z_snk z_src':
        for j, snk in enumerate(snksrc['snks']):
            src = snksrc['srcs'][0]  # assume single source for now
            k = key+'_'+snk+src
            lbl = r'$z_{\rm %s}$' % snk
            eff = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t = np.arange(eff.shape[0])

            if 'exp' in mtype:
                # we have to cut the final t-slice
                zeff = np.exp(eff * t) * dsets[k][:-1]
            elif mtype == 'cosh':
                zeff = dsets[k][:-1] / \
                    (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            # we don't want the last entry (wrap around effects)
            #zeff = zeff[:-1]
            if snk == src:
                z = [d.mean for d in np.sqrt(zeff)]
                dz = [d.sdev for d in np.sqrt(zeff)]
            else:
                k2 = key+'_'+src+src
                eff2 = effective_mass(dsets[k2], mtype=mtype, tau=tau)
                if 'exp' in mtype:
                    zeff2 = np.exp(eff*t) * dsets[k2][:-1]
                elif mtype == 'cosh':
                    zeff2 = dsets[k2][:-1] / \
                        (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
                #zeff2 = zeff2[:-1]
                z = [d.mean for d in zeff / np.sqrt(zeff2)]
                dz = [d.sdev for d in zeff / np.sqrt(zeff2)]

            if colors is not None:
                axes[j].errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            color=colors[snk+src], mfc='None', label=lbl)
            else:
                axes[j].errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            mfc='None', label=lbl)


def plot_stability(fits, tmin, n_states, tn_opt, state,
                   ylim=None, diff=False, save=True, n_plot=0, scale=None):
    fs = 20
    fs_ns = 16
    nn = str(n_plot)
    if n_plot == 0:
        kk = 'E'
    else:
        kk = 'dE'

    markers = dict()
    colors = dict()
    markers[1] = 'o'
    colors[1] = 'r'
    markers[2] = 's'
    colors[2] = 'orange'
    markers[3] = '^'
    colors[3] = 'g'
    markers[4] = 'D'
    colors[4] = 'b'
    markers[5] = '*'
    colors[5] = 'darkviolet'
    markers[6] = 'h'
    colors[6] = 'violet'
    markers[7] = 'X'
    colors[7] = 'gold'
    markers[8] = '8'
    colors[8] = 'darkred'

    fig = plt.figure(state+'_E_'+nn+'_stability', figsize=(7, 4))
    if scale:
        ax_e0 = plt.axes([0.14, 0.42, 0.75, 0.57])
        ax_Q = plt.axes([0.14, 0.27, 0.75, 0.15])
        ax_w = plt.axes([0.14, 0.12, 0.75, 0.15])
        s = float(scale[0])
        units = scale[1]
    else:
        ax_e0 = plt.axes([0.14, 0.42, 0.85, 0.57])
        ax_Q = plt.axes([0.14, 0.27, 0.85, 0.15])
        ax_w = plt.axes([0.14, 0.12, 0.85, 0.15])

    for ti in tmin:
        logGBF = []
        for ns in n_states:
            if n_plot <= ns-1:
                if ti == tmin[0]:
                    if ns == n_states[0]:
                        lbl = r'$n_s=%d$' % ns
                    else:
                        lbl = r'$%d$' % ns
                else:
                    lbl = ''
                if ns == tn_opt[1] and ti == tn_opt[0]:
                    mfc = 'k'
                    color = colors[ns]
                else:
                    mfc = 'None'
                    color = colors[ns]
                logGBF.append(fits[(ti, ns)].logGBF)
                if diff:
                    e0 = fits[(ti, ns)].p[state+'_'+kk+'_'+nn] - fits[(ti, ns)
                                                                      ].p['pi_'+kk+'_'+nn] - fits[(ti, ns)].p['D_'+kk+'_'+nn]
                else:
                    e0 = fits[(ti, ns)].p[state+'_E_0']
                    if n_plot > 0:
                        for i_n in range(1, n_plot+1):
                            e0 += fits[(ti, ns)].p[state+'_'+kk+'_'+str(i_n)]
                ax_e0.errorbar(ti + 0.1*(ns-5), e0.mean, yerr=e0.sdev,
                               marker=markers[ns], color=color, mfc=mfc, linestyle='None', label=lbl)
                ax_Q.plot(ti + 0.1*(ns-5), fits[(ti, ns)].Q,
                          marker=markers[ns], color=color, mfc=mfc, linestyle='None')
        logGBF = np.array(logGBF)
        logGBF = logGBF - logGBF.max()
        weights = np.exp(logGBF)
        weights = weights / weights.sum()
        for i_s, ns in enumerate(n_states):
            if i_s < len(weights):
                if ns == tn_opt[1] and ti == tn_opt[0]:
                    mfc = 'k'
                    color = colors[ns]
                else:
                    mfc = 'None'
                    color = colors[ns]
                ax_w.plot(ti + 0.1*(ns-5), weights[i_s],
                          marker=markers[ns], color=color, mfc=mfc, linestyle='None')
    priors = fits[(tn_opt[0], tn_opt[1])].prior
    if diff:
        ax_e0.set_ylabel(r'$E^{\rm %s}_%s - M_D - M_\pi$' %
                         (state.replace('_', '\_'), nn), fontsize=fs)
        e0_opt = fits[(tn_opt[0], tn_opt[1])].p[state+'_'+kk+'_'+nn]
        e0_opt += -fits[(tn_opt[0], tn_opt[1])].p['pi_'+kk+'_'+nn] - \
            fits[(tn_opt[0], tn_opt[1])].p['D_'+kk+'_'+nn]
        ax_e0.set_ylim(-2*abs(e0.mean), 2*abs(e0.mean))
        ax_e0.axhline(0)
    else:
        ax_e0.set_ylabel(r'$E^{\rm %s}_%s$' % (state, nn), fontsize=fs)
        e0_opt = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
        if n_plot == 0:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].prior[state+'_E_0']
            ax_e0.set_ylim(
                1.001*(e0_prior.mean-e0_prior.sdev),
                0.999*(e0_prior.mean+e0_prior.sdev))
        elif n_plot > 0 and n_plot < tn_opt[1]:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
            for i_n in range(1, n_plot):
                e0_prior += fits[(tn_opt[0], tn_opt[1])
                                 ].p[state+'_'+kk+'_'+str(i_n)]
                e0_opt += fits[(tn_opt[0], tn_opt[1])
                               ].p[state+'_'+kk+'_'+str(i_n)]
            e0_prior += fits[(tn_opt[0], tn_opt[1])
                             ].prior[state+'_'+kk+'_'+str(n_plot)]
            e0_opt += fits[(tn_opt[0], tn_opt[1])].p[state
                                                     + '_'+kk+'_'+str(n_plot)]
        else:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
        if n_plot > 0 and n_plot < tn_opt[1]:
            ax_e0.set_ylim(
                0.75*(e0_prior.mean-e0_prior.sdev),
                1.25*(e0_prior.mean+e0_prior.sdev))
            ax_e0.axhline(e0_prior.mean-e0_prior.sdev,
                          linestyle='--', color='k', alpha=.5)
            ax_e0.axhline(e0_prior.mean+e0_prior.sdev,
                          linestyle='--', color='k', alpha=.5)

    if scale:
        ax_e0r = ax_e0.twinx()
        ax_e0r.set_ylim(ax_e0.get_ylim()[0]*s, ax_e0.get_ylim()[1]*s)
        ax_e0r.set_yticks([s*t for t in ax_e0.get_yticks()[1:-1]])
        if units in ['GeV', 'gev']:
            ax_e0r.set_yticklabels(["%.2f" % t for t in ax_e0r.get_yticks()])
        else:
            ax_e0r.set_yticklabels(["%.0f" % t for t in ax_e0r.get_yticks()])

    #ax_e0.text(0.05,0.1, text, transform=ax_e0.transAxes,
    #    bbox={'facecolor':ens_colors[a_str],'boxstyle':'round'},
    #    horizontalalignment='left', fontsize=fs)
    ax_e0.legend(loc=1, ncol=8, columnspacing=0,
                 fontsize=fs_ns, handletextpad=0.1)
    if n_plot >= 0 and n_plot < tn_opt[1]:
        ax_e0.axhspan(e0_opt.mean-e0_opt.sdev, e0_opt.mean
                      + e0_opt.sdev, color=colors[tn_opt[1]], alpha=.2)
    ax_e0.set_xticks(tmin)
    if ylim is not None:
        ax_e0.set_ylim(ylim)

    ax_Q.set_xticks(tmin)
    ax_Q.set_yticks([0.1, 0.75])
    ax_w.set_xticks(tmin)
    ax_w.set_yticks([0.1, 0.75])
    ax_e0.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_Q.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_w.tick_params(bottom=True, top=True, direction='in')

    ax_w.set_ylabel(r'$w_{n_s}$', fontsize=fs)
    ax_w.set_xlabel(r'$t_{\rm min}$', fontsize=fs)
    ax_w.set_ylim(0, 1.2)
    ax_Q.set_ylabel(r'$Q$', fontsize=fs)
    ax_Q.set_ylim(0, 1.2)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+state+'_E_'+nn+'_stability.pdf', transparent=True)
