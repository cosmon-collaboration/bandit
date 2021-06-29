import numpy as np

'''
x['pi_SS'] = {'state':'pi'
              'corr':cosh, 'n_state':3,
              't_range':np.arange(10,21,1), 'T':T
              'snk':'S', 'src':'S'
              'factorized_fit (z_snk_n z_src_n) or not (A_n)'
              }
'''

class CorrFunction:

    def En(self,x,p,n):
        '''
        g.s. energy is energy
        e.s. energies are given as dE_n = E_n - E_{n-1}
        '''
        E = p['%s_E_0' %x['state']]
        for i in range(1,n+1):
            E += p['%s_dE_%s' %(x['state'], i)]
        return E

    def exp(self, x, p):
        r = 0
        t = x['t_range']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" %(x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" %(x['state'], x['snk'], n)]
            E_n = self.En(x,p,n)
            r +=  z_snk * z_src * np.exp(-E_n*t)
        return r

    def cosh(self, x, p):
        r = 0
        t = x['t_range']
        T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" %(x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" %(x['state'], x['snk'], n)]
            E_n = self.En(x,p,n)
            r += z_snk * z_src * np.exp(-E_n*t)
            r += z_snk * z_src * np.exp(-E_n*(T-t))
        return r

    def cosh_const(self, x, p):
        r = 0
        t = x['t_range']
        T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" %(x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" %(x['state'], x['snk'], n)]
            E_n = self.En(x,p,n)
            r += z_snk * z_src * np.exp(-E_n*t)
            r += z_snk * z_src * np.exp(-E_n*(T-t))
        # add the "const" term
        z_src = p["%s_z%s_c_0" %(x['state'], x['src'])]
        z_snk = p["%s_z%s_c_0" %(x['state'], x['snk'])]
        r += z_src * z_snk * np.exp(-p['D_E_0']*t) * np.exp(-p['pi_E_0']*(T-t))
        r += z_src * z_snk * np.exp(-p['pi_E_0']*t) * np.exp(-p['D_E_0']*(T-t))

        return r

    def eff_mass(self, x, p, ax, color='k', alpha=.4, tau=1):
        xp = dict(x)
        xm = dict(x)
        xp['t_range'] = x['t_range']+tau
        xm['t_range'] = x['t_range']-tau
        if x['type'] in ['cosh', 'cosh_const']:
            if x['type'] == 'cosh':
                corr   = self.cosh(x, p)
                corr_p = self.cosh(xp,p)
                corr_m = self.cosh(xm,p)
            else:
                corr   = self.cosh_const(x, p)
                corr_p = self.cosh_const(xp,p)
                corr_m = self.cosh_const(xm,p)
            meff = 1/tau * np.arccosh( (corr_p + corr_m) / 2 / corr )
        elif x['type'] == 'exp':
            corr   = self.exp(x, p)
            corr_p = self.exp(xp,p)
            meff = 1/tau * np.log( corr / corr_p )
        m  = np.array([k.mean for k in meff])
        dm = np.array([k.sdev for k in meff])
        ax.fill_between(x['t_range'], m-dm, m+dm,color=color, alpha=alpha)


class FitCorr(object):
    def __init__(self):
        self.corr_functions = CorrFunction()

    def priors(self,prior):
        '''
        only keep priors that are used in fit
        truncate based on n_states
        '''
        p = dict()
        for q in prior:
            if 'log' in q:
                if int(q.split('_')[-1].split(')')[0]) < self.nstates:
                    p[q] = prior[q]
            else:
                if int(q.split('_')[-1]) < self.nstates:
                    p[q] = prior[q]
        return p

    def fit_function(self,x,p):
        r = dict()
        for k in x:
            if x[k]['type'] == 'exp':
                r[k] = self.corr_functions.exp(x[k],p)
            elif x[k]['type'] == 'cosh':
                r[k] = self.corr_functions.cosh(x[k],p)
            elif x[k]['type'] == 'cosh_const':
                r[k] = self.corr_functions.cosh_const(x[k],p)
        return r
