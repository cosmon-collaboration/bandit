import numpy as np
import sys
import gvar as gv 
import lsqfit
import matplotlib.pyplot as plt


import two_pt_fit as two_pt 
import three_pt_fit as three_pt 
import corr_functions as corr


class Fit_Collect(object):

    def __init__(self, n_states, prior, t_range,
                 hadron_corr_gv=None, axial_fh_num_gv=None, vector_fh_num_gv=None):

        self.n_states = n_states
        self.t_range = t_range
        self.prior = prior
        self.hadron_corr_gv = hadron_corr_gv
        self.axial_fh_num_gv = axial_fh_num_gv
        self.vector_fh_num_gv = vector_fh_num_gv
        self.fit = None
        self.prior = self._make_prior(prior)


    def get_fit(self):
        if self.fit is not None:
            return self.fit
        else:
            return self._make_fit()

    def get_energies(self):
        # Don't rerun the fit if it's already been made
        if self.fit is not None:
            temp_fit = self.fit
        else:
            temp_fit = self.get_fit()

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        output = gv.gvar(np.zeros(max_n_states))
        output[0] = temp_fit.p['E0']
        for k in range(1, max_n_states):
            output[k] = output[0] + np.sum([(temp_fit.p['dE'][j]) for j in range(k)], axis=0)
        return output

    def _make_fit(self):
        '''
        1. create a model (subclass of lsqfit.MultiFitter)
        2. make a fitter using the models
        3. Perform the fit with chosen set of correlators
        '''

        models = self._make_models_simult_fit()

        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=self._make_data(), prior=self.prior)
        self.fit = fit
        return fit

    def _make_models_simult_fit(self):
        models = np.array([])

        if self.hadron_corr_gv is not None:
            for sink in list(self.hadron_corr_gv.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    # 'E1'      : 'E1',
                    # 'E2'      : 'E2',
                    # 'E3'      : 'E3',
                    'log(dE)' : 'log(dE)',
                    'wf'      : 'wf_'+sink,
                }
                models = np.append(models,
                           baryon_model(datatag="hadron_"+sink,
                           t=list(range(self.t_range['corr'][0], self.t_range['corr'][1])),
                           param_keys=param_keys, n_states=self.n_states['corr']))

        if self.axial_fh_num_gv is not None:
            for sink in list(self.axial_fh_num_gv.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    'log(dE)' : 'log(dE)',
                    'd'       : 'd_A_'+sink,
                    'g_nm'    : 'g_A_nm',
                    'wf'      : 'wf_'+sink,
                }
                models = np.append(models,
                           fh_num_model(datatag="axial_fh_num_"+sink,
                           t=list(range(self.t_range['gA'][0], self.t_range['gA'][1])),
                           param_keys=param_keys, n_states=self.n_states['gA']))

        if self.vector_fh_num_gv is not None:
            for sink in list(self.vector_fh_num_gv.keys()):
                param_keys = {
                    'E0'      : 'E0',
                    'log(dE)' : 'log(dE)',
                    'd'       : 'd_V_'+sink,
                    'g_nm'    : 'g_V_nm',
                    'wf'      : 'wf_'+sink,
                }
                models = np.append(models,
                           fh_num_model(datatag="vector_fh_num_"+sink,
                           t=list(range(self.t_range['gV'][0], self.t_range['gV'][1])),
                           param_keys=param_keys, n_states=self.n_states['gV']))

        return models

    # data array needs to match size of t array
    def _make_data(self):
        data = {}
        if self.hadron_corr_gv is not None:
            for sink in list(self.hadron_corr_gv.keys()):
                data["hadron_"+sink] = self.hadron_corr_gv[sink][self.t_range['corr'][0]:self.t_range['corr'][1]]

        if self.axial_fh_num_gv is not None:
            for sink in list(self.axial_fh_num_gv.keys()):
                data["axial_fh_num_"+sink] = self.axial_fh_num_gv[sink][self.t_range['gA'][0]:self.t_range['gA'][1]]

        if self.vector_fh_num_gv is not None:
            for sink in list(self.vector_fh_num_gv.keys()):
                data["vector_fh_num_"+sink]  = self.vector_fh_num_gv[sink][self.t_range['gV'][0]:self.t_range['gV'][1]]

        return data

    def _make_prior(self, prior):
        resized_prior = {}

        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        for key in list(prior.keys()):
            if key == 'g_A_nm':
                resized_prior[key] = prior[key][:self.n_states['gA'], :self.n_states['gA']]
            elif key == 'g_V_nm':
                resized_prior[key] = prior[key][:self.n_states['gV'], :self.n_states['gV']]
            elif key in ['d_A_dir', 'd_A_smr']:
                resized_prior[key] = prior[key][:self.n_states['gA']]
            elif key in ['d_V_dir', 'd_V_smr']:
                resized_prior[key] = prior[key][:self.n_states['gV']]
            else:
                resized_prior[key] = prior[key][:max_n_states]

        new_prior = resized_prior.copy()
        new_prior['E0'] = resized_prior['E'][0]
        # Don't need this entry
        new_prior.pop('E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
        new_prior['log(dE)'] = gv.gvar(np.zeros(len(resized_prior['E']) - 1))
        for j in range(len(new_prior['log(dE)'])):
            #excited_state_energy = p[self.mass] + np.sum([np.exp(p[self.log_dE][k]) for k in range(j-1)], axis=0)

            # Notice that I've coded this s.t.
            # the std is determined entirely by the excited state
            # dE_mean = gv.mean(resized_prior['E'][j+1] - resized_prior['E'][j])
            # dE_std = gv.sdev(resized_prior['E'][j+1])
            temp = gv.gvar(resized_prior['E'][j+1]) - gv.gvar(resized_prior['E'][j])
            temp2 = gv.gvar(resized_prior['E'][j+1])
            temp_gvar = gv.gvar(temp.mean,temp2.sdev)
            new_prior['log(dE)'][j] = np.log(temp_gvar)

        return new_prior


        max_n_states = np.max([self.n_states[key] for key in list(self.n_states.keys())])
        output = gv.gvar(np.zeros(max_n_states))
        output[0] = temp_fit.p['E0']
        for k in range(1, max_n_states):
            output[k] = output[0] + np.sum([(temp_fit.p['dE'][j]) for j in range(k)], axis=0)
        return output
