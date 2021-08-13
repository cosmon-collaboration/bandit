import matplotlib.pyplot as plt
import numpy as np
import os


def effective_mass(gvdata, mtype='exp', tau=1):
    if mtype == 'exp':
        meff = 1./tau * np.log(gvdata / np.roll(gvdata,-tau))
    elif mtype == 'cosh':
        meff = 1./tau * np.arccosh((np.roll(gvdata,-tau)+np.roll(gvdata,tau))/2/gvdata)
    # chop off last time slice for wrap-around effects
    return meff[:-1]

def plot_eff(ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0):
    lst = [k for k in dsets if key in k]
    for k in lst:
        lbl = k.split('_')[1]
        eff = effective_mass(dsets[k], mtype=mtype, tau=tau)
        x   = np.arange(eff.shape[0] + offset)
        m   = [k.mean for k in eff]
        dm  = [k.sdev for k in eff]
        if colors is not None:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                color=colors[lbl], mfc='None', label=lbl)
        else:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                mfc='None', label=label)

def plot_zeff(ax, dsets, key, ztype='A_snk,src', snksrc=None, mtype='exp', tau=1, colors=None):
    lst = [k for k in dsets if key in k]
    if ztype == 'A_snk,src':
        for k in lst:
            lbl  = k.split('_')[-1]
            eff  = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t    = np.arange(eff.shape[0])
            if mtype=='exp':
                zeff = np.exp(eff * t) * dsets[k]
            elif mtype=='cosh':
                zeff = dsets[k] / (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            z    = [k.mean for k in zeff]
            dz   = [k.sdev for k in zeff]
            if colors is not None:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                    color=colors[label], mfc='None', label=lbl)
            else:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                    mfc='None', label=lbl)
    elif ztype == 'z_snk z_src':
        for snk in snksrc['snks']:
            src  = snksrc['srcs'][0]# assume single source for now
            k    = key+'_'+snk+src
            lbl  = r'$z_{\rm %s}$' %snk
            eff  = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t    = np.arange(eff.shape[0])

            if mtype=='exp':
                zeff = np.exp(eff * t) * dsets[k][:-1]# we have to cut the final t-slice
            elif mtype=='cosh':
                zeff = dsets[k][:-1] / (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            # we don't want the last entry (wrap around effects)
            #zeff = zeff[:-1]
            if snk == src:
                z  = [d.mean for d in np.sqrt(zeff)]
                dz = [d.sdev for d in np.sqrt(zeff)]
            else:
                k2   = key+'_'+src+src
                eff2 = effective_mass(dsets[k2], mtype=mtype, tau=tau)
                if mtype=='exp':
                    zeff2 = np.exp(eff*t) * dsets[k2][:-1]
                elif mtype=='cosh':
                    zeff2 = dsets[k2][:-1] / (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
                #zeff2 = zeff2[:-1]
                z     = [d.mean for d in zeff / np.sqrt(zeff2) ]
                dz    = [d.sdev for d in zeff / np.sqrt(zeff2) ]

            if colors is not None:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                    color=colors[snk+src], mfc='None', label=lbl)
            else:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                    mfc='None', label=lbl)

def plot_stability(fits, tmin, n_states, tn_opt, state, ylim=None, diff=False, save=True):
    fs = 20
    fs_ns = 16

    markers = dict(); colors = dict()
    markers[1] = 'o'; colors[1] = 'r'
    markers[2] = 's'; colors[2] = 'orange'
    markers[3] = '^'; colors[3] = 'g'
    markers[4] = 'D'; colors[4] = 'b'
    markers[5] = '*'; colors[5] = 'darkviolet'
    markers[6] = 'h'; colors[6] = 'violet'
    markers[7] = 'X'; colors[7] = 'gold'
    markers[8] = '8'; colors[8] = 'darkred'

    fig   = plt.figure(state+'_stability', figsize=(7,4))
    ax_e0 = plt.axes([0.14,0.42,0.85,0.57])
    ax_Q  = plt.axes([0.14,0.27,0.85,0.15])
    ax_w  = plt.axes([0.14,0.12,0.85,0.15])

    for ti in tmin:
        logGBF = []
        for ns in n_states:
            if ti == tmin[0]:
                if ns == n_states[0]:
                    lbl = r'$n_s=%d$' %ns
                else:
                    lbl = r'$%d$' %ns
            else:
                lbl = ''
            if ns == tn_opt[1] and ti == tn_opt[0]:
                mfc = 'k'; color=colors[ns]
            else:
                mfc = 'None'; color=colors[ns]
            logGBF.append(fits[(ti,ns)].logGBF)
            if diff:
                e0 = fits[(ti,ns)].p[state+'_E_0'] - fits[(ti,ns)].p['pi_E_0'] - fits[(ti,ns)].p['D_E_0']
            else:
                e0 = fits[(ti,ns)].p[state+'_E_0']
            ax_e0.errorbar(ti +0.1*(ns-5), e0.mean, yerr=e0.sdev,
                marker=markers[ns], color=color, mfc=mfc,linestyle='None',label=lbl)
            ax_Q.plot(ti +0.1*(ns-5), fits[(ti,ns)].Q,
                marker=markers[ns], color=color, mfc=mfc,linestyle='None')
        logGBF = np.array(logGBF)
        logGBF = logGBF - logGBF.max()
        weights = np.exp(logGBF)
        weights = weights / weights.sum()
        for i_s,ns in enumerate(n_states):
            if ns == tn_opt[1] and ti == tn_opt[0]:
                mfc = 'k'; color=colors[ns]
            else:
                mfc = 'None'; color=colors[ns]
            ax_w.plot(ti +0.1*(ns-5), weights[i_s],
                marker=markers[ns], color=color, mfc=mfc,linestyle='None')
    priors = fits[(tmin[0],n_states[0])].prior
    if diff:
        ax_e0.set_ylabel(r'$E^{\rm %s}_0 - M_D - M_\pi$' %(state.replace('_','\_')), fontsize=fs)
        e0_opt  = fits[(tn_opt[0],tn_opt[1])].p[state+'_E_0']
        e0_opt += -fits[(tn_opt[0],tn_opt[1])].p['pi_E_0'] -fits[(tn_opt[0],tn_opt[1])].p['D_E_0']
        ax_e0.set_ylim(-2*abs(e0.mean), 2*abs(e0.mean))
        ax_e0.axhline(0)
    else:
        ax_e0.set_ylabel(r'$E^{\rm %s}_0$' %state, fontsize=fs)
        e0_opt = fits[(tn_opt[0],tn_opt[1])].p[state+'_E_0']
        ax_e0.set_ylim(
            1.001*priors[state+'_E_0'].mean-priors[state+'_E_0'].sdev,
            0.999*priors[state+'_E_0'].mean+priors[state+'_E_0'].sdev)
    #ax_e0.text(0.05,0.1, text, transform=ax_e0.transAxes,
    #    bbox={'facecolor':ens_colors[a_str],'boxstyle':'round'},
    #    horizontalalignment='left', fontsize=fs)
    ax_e0.legend(loc=1, ncol=8, columnspacing=0, fontsize=fs_ns, handletextpad=0.1)
    ax_e0.axhspan(e0_opt.mean-e0_opt.sdev, e0_opt.mean+e0_opt.sdev, color=colors[tn_opt[1]], alpha=.2)
    ax_e0.set_xticks(tmin)
    if ylim is not None:
        ax_e0.set_ylim(ylim)

    ax_Q.set_xticks(tmin)
    ax_Q.set_yticks([0.1,0.75])
    ax_w.set_xticks(tmin)
    ax_w.set_yticks([0.1,0.75])
    ax_e0.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_Q.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_w.tick_params(bottom=True, top=True, direction='in')

    ax_w.set_ylabel(r'$w_{n_s}$', fontsize=fs)
    ax_w.set_xlabel(r'$t_{\rm min}$', fontsize=fs)
    ax_w.set_ylim(0,1.2)
    ax_Q.set_ylabel(r'$Q$', fontsize=fs)
    ax_Q.set_ylim(0,1.2)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+state+'_stability.pdf', transparent=True)
