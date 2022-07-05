import numpy as np
import sys

'''
x['pi_SS'] = {'state':'pi'
              'corr':cosh, 'n_state':3,
              't_range':np.arange(10,21,1), 'T':T
              'snk':'S', 'src':'S'
              'factorized_fit (z_snk_n z_src_n) or not (A_n)'
              }
'''


class CorrFunction:
    def mres_func(self,x,p):
        return p['mres']*np.ones(x.shape[0])

    def En(self, x, p, n):
        '''
        g.s. energy is energy
        e.s. energies are given as dE_n = E_n - E_{n-1}
        '''
        E = p['%s_E_0' % x['state']]
        for i in range(1, n+1):
            E += p['%s_dE_%d' % (x['state'], i)]
        return E

    def dEn(self, x, p, n):
        E = p['%s_dE_1' % x['state']]
        for i in range(2, n+1):
            E += p['%s_dE_%d' % (x['state'], i)]
        return E

    def E_el_n(self, x, p, n):
        E = p['%s_E_el_1' % x['state']]
        for i in range(2, n+1):
            E += p['%s_E_el_%d' % (x['state'], i)]
        return E

    def exp(self, x, p):
        #print('DEBUG:',x)
        #sys.exit()
        r = 0
        t = x['t_range']
        for n in range(x['n_state']):
            E_n = self.En(x, p, n)
            if x['ztype'] == 'z_snk z_src':
                z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
                z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
                r += z_snk * z_src * np.exp(-E_n*t)
            elif x['ztype'] == 'A_snk,src':
                A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'], n)]
                r += A * np.exp(-E_n*t)
        return r

    def exp_open(self, x, p):
        ''' first add exp tower '''
        r = self.exp(x,p)
        ''' then add first open boundary condition term
            r += A_open * exp( -(E_2pi - m_pi)*(T-t) )

            we will model E_2pi = m_pi + dEOpen
            therefore, we can set E = dEOpen
        '''
        t = x['t_range']
        T = x['T']
        E = p['%s_dEOpen_1' % (x['state'])]
        if x['ztype'] == 'z_snk z_src':
            z_src = p["%s_z%sOpen_1" % (x['state'], x['src'])]
            z_snk = p["%s_z%sOpen_1" % (x['state'], x['snk'])]
            r += z_snk * z_src * np.exp(-E *(T-t))
        elif x['ztype'] == 'A_snk,src':
            A = p['%s_z%s%sOpen_1' % (x['state'], x['snk'], x['src'])]
            r += A * np.exp(-E *(T-t))
        return r

    def exp_gs(self,x,p):
        ''' model the correlator with a geometric series inspired model

            C(t) = A0 exp(-E0 t) / (1 - sum_n A_n exp( -dEn t))
        '''
        t  = x['t_range']
        E0 = self.En(x, p, 0)
        if x['ztype'] == 'z_snk z_src':
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], 0)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], 0)]
            r = z_snk * z_src * np.exp( -E0 * t)
        elif x['ztype'] == 'A_snk,src':
            A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'], 0)]
            r = A * np.exp(-E0*t)
        ''' build denominator '''
        r_den = 1
        for n in range(1, x['n_state']):
            dEn = self.dEn(x, p, n)
            if x['ztype'] == 'z_snk z_src':
                z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
                z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
                r_den += -z_snk * z_src * np.exp(-dEn*t)
            elif x['ztype'] == 'A_snk,src':
                A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'], n)]
                r_den += -A * np.exp(-dEn*t)
        return r / r_den

    def cosh(self, x, p):
        r = 0
        t = x['t_range']
        T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            E_n = self.En(x, p, n)
            r += z_snk * z_src * (np.exp(-E_n*t) + np.exp(-E_n*(T-t)))
        return r
    
    def mres(self, x, p):
        ''' m_res = midpoint_pseudo / pseudo_pseudo
            we fit to a constant away from early/late time
            m_res = p[mres_l]
        '''
        return p[x['state']]*np.ones_like(x['t_range'])
    

    def mres(self, x, p):
        ''' m_res = midpoint_pseudo / pseudo_pseudo
            we fit to a constant away from early/late time
            m_res = p[mres_l]
        '''
        return p[x['state']]*np.ones_like(x['t_range'])

    def two_h_ratio(self, x, p):
        ''' This model fits the g.s. as

            E_hh_0 = E1_0 + E2_0 + Delta_00

            and all other states independently of the single hadron energy levels.
        '''
        t = x['t_range']
        z_src = p["%s_z%s_%d" % (x['state'], x['src'], 0)]
        z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], 0)]
        E0 = p[x['denom'][0]+'_E_0'] + p[x['denom']
                                         [1]+'_E_0'] + p['%s_dE_0_0' % x['state']]
        r = z_snk * z_src * np.exp(-E0*t)
        for n in range(1, x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            En = E0 + self.dEn(x, p, n)
            r += z_snk * z_src * np.exp(-En*t)
        return r

    def two_h_conspire(self, x, p):
        ''' This model assumes strong cancelation of e.s. in two and single hadron
            correlation functions.  Therefore, it uses many priors from the single
            hadron states as input.  For example, the correlator model looks like

            C(t) = A_0**2 * exp(-(2E_0+D_00)*t) + 2A0*A1*exp(-(E0+E1+D_01)*t) + A1**2*exp(-(2E1+D_11)*t)+...

            In addition, we have the option to add extra "elastic" excited states
        '''
        r = 0
        t = x['t_range']
        for n in range(x['n_state']):
            for m in range(x['n_state']):
                if n <= m:
                    En = self.En({'state': x['denom'][0]}, p, n)
                    Em = self.En({'state': x['denom'][1]}, p, m)
                    Dnm = p['%s_dE_%d_%d' % (x['state'], n, m)]
                    Anm = p['%s_A%s_%d_%d' %
                            (x['state'], x['snk']+x['src'], n, m,)]
                    #print(x['state'],n,m,En,Em,Dnm,Anm)
                    if n == m:
                        r += Anm * np.exp(-(En+Em+Dnm)*t)
                    elif n < m:
                        r += 2*Anm * np.exp(-(En+Em+Dnm)*t)
        if 'n_el' in x and x['n_el'] > 0:
            for n in range(1, x['n_el']+1):
                E_el_n = p['%s_E_0' % x['denom'][0]]
                E_el_n += p['%s_E_0' % x['denom'][0]]
                E_el_n += self.E_el_n(x, p, n)
                z_src = p["%s_z%s_el_%d" % (x['state'], x['src'], n)]
                z_snk = p["%s_z%s_el_%d" % (x['state'], x['snk'], n)]
                r += z_snk * z_src * np.exp(-E_el_n*t)
        return r

    def cosh_const(self, x, p):
        r = 0
        t = x['t_range']
        T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            E_n = self.En(x, p, n)
            r += z_snk * z_src * np.exp(-E_n*t)
            r += z_snk * z_src * np.exp(-E_n*(T-t))
        # add the "const" term
        z_src = p["%s_z%s_c_0" % (x['state'], x['src'])]
        z_snk = p["%s_z%s_c_0" % (x['state'], x['snk'])]
        r += z_src * z_snk * np.exp(-p['D_E_0']*t) * np.exp(-p['pi_E_0']*(T-t))
        r += z_src * z_snk * np.exp(-p['pi_E_0']*t) * np.exp(-p['D_E_0']*(T-t))

        return r

    def eff_mass(self, x, p, ax, color='k', alpha=.4, t0=0, tau=1, denom_x=None):
        xp = dict(x)
        xm = dict(x)
        xp['t_range'] = x['t_range']+tau
        xm['t_range'] = x['t_range']-tau
        if x['type'] in ['cosh', 'cosh_const']:
            if x['type'] == 'cosh':
                corr = self.cosh(x, p)
                corr_p = self.cosh(xp, p)
                corr_m = self.cosh(xm, p)
            else:
                corr = self.cosh_const(x, p)
                corr_p = self.cosh_const(xp, p)
                corr_m = self.cosh_const(xm, p)
            meff = 1/tau * np.arccosh((corr_p + corr_m) / 2 / corr)
        elif 'exp' in x['type']:
            if denom_x:
                x0 = dict(denom_x[0])
                x1 = dict(denom_x[1])
                x0p = dict(x0)
                x1p = dict(x1)
                x0p['t_range'] = x0['t_range']+tau
                x1p['t_range'] = x1['t_range']+tau
                if x['type'] == 'exp_r':
                    corr = self.two_h_ratio(x, p) / self.exp(x0, p) / self.exp(x1, p)
                    corr_p = self.two_h_ratio(xp, p) / self.exp(x0p, p) / self.exp(x1p, p)
                elif x['type'] == 'exp_r_conspire':
                    corr = self.two_h_conspire(x, p) / self.exp(x0, p) / self.exp(x1, p)
                    corr_p = self.two_h_conspire(xp, p) / self.exp(x0p, p) / self.exp(x1p, p)
                elif x['type'] == 'exp_r_ind':
                    corr = self.exp(x, p)
                    corr_p = self.exp(xp, p)
            elif x['type'] == 'exp_r':
                corr = self.two_h_ratio(x, p)
                corr_p = self.two_h_ratio(xp, p)
            elif x['type'] == 'exp_r_conspire':
                corr = self.two_h_conspire(x, p)
                corr_p = self.two_h_conspire(xp, p)
            elif x['type'] in ['exp', 'exp_r_ind']:
                corr = self.exp(x, p)
                corr_p = self.exp(xp, p)
            elif x['type'] == 'exp_open':
                corr = self.exp_open(x, p)
                corr_p = self.exp_open(xp, p)
            elif x['type'] == 'exp_gs':
                corr = self.exp_gs(x, p)
                corr_p = self.exp_gs(xp, p)
            meff = 1/tau * np.log(corr / corr_p)
        elif x['type'] == 'mres':
            corr = self.mres(x,p)
            meff = corr
        else:
            sys.exit('unrecognized type, %s' % x['type'])

        m = np.array([k.mean for k in meff])
        dm = np.array([k.sdev for k in meff])
        ax.fill_between(x['t_range']+t0, m-dm, m+dm, color=color, alpha=alpha)


class FitCorr(object):
    def __init__(self):
        self.corr_functions = CorrFunction()

    def get_fit(self,priors,states,x,y):
        p0 = {k: v.mean for (k, v) in priors.items()}
        # only pass x for states in fit
        x_fit = dict()
        y_fit = dict()
        fit_lst = [k for k in x if k.split('_')[0] in states]
        for k in fit_lst:
            if 'mres' not in k:
                x_fit[k] = x[k]
                y_fit[k] = y[k]
            else:
                k_res = k.split('_')[0]
                if k_res not in x_fit:
                    x_fit[k_res] = x[k]
                    y_fit[k_res] = y[k_res]
        return p0,x_fit,y_fit

    def priors(self, prior):
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

    def fit_function(self, x, p):
        r = dict()
        for k in x:
            if x[k]['type'] in dir(self.corr_functions):
                r[k] = getattr(self.corr_functions, x[k]['type'])(x[k],p)
            # NOTE: we should move exp_r and exp_r_conspire into corr_functions
            elif x[k]['type'] == 'exp_r':
                sp = k.split('_')[-1]
                r[k] = self.corr_functions.two_h_ratio(x[k], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][0]+'_'+sp], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][1]+'_'+sp], p)
            elif x[k]['type'] == 'exp_r_conspire':
                sp = k.split('_')[-1]
                r[k] = self.corr_functions.two_h_conspire(x[k], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][0]+'_'+sp], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][1]+'_'+sp], p)
            else:
                sys.exit('Unrecognized fit model, %s' %x[k]['type'])
        return r
