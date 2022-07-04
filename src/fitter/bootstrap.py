import numpy as np
from numpy.random import Generator, SeedSequence, PCG64
import hashlib
import h5py
import sys,os
import gvar as gv
import lsqfit

#local modules
import corr_functions as cf
fit_funcs = cf.FitCorr()

class Bootstrap(object):
    def __init__(self):
        # input args to fit_twopt
        self.bs_results = bs_results
        self.bs_path    = bs_path
        self.overwrite  = overwrite
        self.Nbs        = Nbs
        self.fit        = fit
        self.fit_lst    = fit_lst
        self.bs_seed    = bs_seed
        self.fp         = fp
        self.verbose    = verbose

    def run_bs(self):
        if not os.path.exists('bs_results'):
            os.makedirs('bs_results')
        if len(bs_results.split('/')) == 1:
            bs_file = 'bs_results/'+self.bs_results
        else:
            bs_file = self.bs_results
        # check if we already wrote this dataset
        have_bs = False
        if os.path.exists(bs_file):
            with h5py.File(bs_file, 'r') as f5:
                if self.bs_path in f5:
                    if len(f5[self.bs_path]) > 0 and not overwrite:
                        have_bs = True
                        print(
                            'you asked to write bs results to an existing dset and overwrite =', self.overwrite)
        if not have_bs:
            print('beginning Nbs=%d bootstrap fits' % Nbs)
        
        p0_bs = dict()
        for k in self.fit.p:
            p0_bs[k] = self.fit.p[k].mean

        if not bs_seed and 'bs_seed' not in dir(self.fp):
            tmp = input('you have not passed a BS seed nor is it defined in the input file\nenter a seed or hit return for none')
            if not tmp:
                self.bs_seed = None
            else:
                self.bs_seed = tmp
        elif 'bs_seed' in dir(fp):
            self.bs_seed = self.fp.bs_seed
        if self.bs_seed:
            if self.verbose:
                print('WARNING: you are overwriting the bs_seed from the input file')
            bs_seed = self.bs_seed
        #return(p0_bs)

    def make_bs_data(self, data_cfg):
        # make BS data
        corr_bs = {}
        for k in data_cfg:
            corr_bs[k] = bs_corrs(data_cfg[k], Nbs=Nbs, seed=bs_seed, return_mbs=True)
        return corr_bs

    def make_bs_lst_priors(self,priors):
        # make BS list for priors
        p_bs_mean = dict()
        for k in priors:
            p_bs_mean[k] = bs_prior(Nbs, mean=priors[k].mean,
                                    sdev=priors[k].sdev, seed=self.bs_seed+'_'+k)

    def make_bs_lst_posterior(self, x_fit,fit_lst,priors):
        post_bs = dict()
        for k in self.fit.p:
            post_bs[k] = []

        for bs in range(self.Nbs):
            sys.stdout.write('%4d / %d\r' % (bs, self.Nbs))
            sys.stdout.flush()

            ''' all gvar's created in this switch are destroyed at restore_gvar [they are out of scope] '''
            gv.switch_gvar()

            bs_data = dict()
            corr_bs = make_bs_data(self,data_cfg)
            for k in corr_bs:
                bs_data[k] = corr_bs[k][bs]
            bs_gv = gv.dataset.avg_data(bs_data)
            if any(['mres' in k for k in bs_gv]):
                bs_tmp = {k:v for (k,v) in bs_gv.items() if 'mres' not in k}
                for k in [key for key in bs_gv if 'mres' in key]:
                    mres = k.split('_')[0]
                    if mres not in bs_tmp:
                        bs_tmp[mres] = bs_gv[mres+'_MP'] / bs_gv[mres+'_PP']
                bs_gv = bs_tmp
            y_bs = {k: v[x_fit[k]['t_range']]
                    for (k, v) in bs_gv.items() if k in self.states}
            p_bs = dict()
            for k in p_bs_mean:
                p_bs[k] = gv.gvar(p_bs_mean[k][bs], priors[k].sdev)
            # do the fit
            if has_svd:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                                fcn=fit_funcs.fit_function, svdcut=svdcut)
            else:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                                fcn=fit_funcs.fit_function)

            for r in post_bs:
                post_bs[r].append(fit_bs.p[r].mean)

            ''' end of gvar scope used for bootstrap '''
            gv.restore_gvar()

        for r in post_bs:
            post_bs[r] = np.array(post_bs[r])
        # write the results
        with h5py.File(bs_file, 'a') as f5:
            try:
                f5.create_group(bs_path)
            except Exception as e:
                print(e)
            for r in post_bs:
                if len(post_bs[r]) > 0:
                    if r in f5[bs_path]:
                        del f5[bs_path+'/'+r]
                    f5.create_dataset(
                        bs_path+'/'+r, data=post_bs[r])

        print('DONE')

    def get_rng(seed: str, verbose=False):
        """Generate a random number generator based on a seed string."""
        # Over python iteration the traditional hash was changed. So, here we fix it to md5
        hash = hashlib.md5(seed.encode("utf-8")).hexdigest()  # Convert string to a hash
        seed_int = int(hash, 16) % (10 ** 6)  # Convert hash to an fixed size integer
        if verbose:
            print("Seed to md5 hash:", seed, "->", hash, "->", seed_int)
        # Create instance of random number generator explicitly to ensure long time support
        # PCG64 -> https://www.pcg-random.org/
        # see https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
        rng = Generator(PCG64(SeedSequence(seed_int)))
        return rng


    def bs_corrs(corr, Nbs, Mbs=None, seed=None, return_bs_list=False, return_mbs=False, verbose=False):
        ''' generate bootstrap resampling of correlation function data
            Args:
                - corr: numpy array of data (Ncfg, Nt, ...)
                - Nbs:  the number of bootstrap samples to generate
                - Mbs:  the number of random draws per bootstrap to generate
                        if Mbs != Ncfg, you will have to appropriately rescale
                        the fluctuations by sqrt( Mbs / Ncfg)
                - seed: a string that will be hashed to seed the random number generator

            Return:
                return_mbs=False
                    corr_bs: an array of shape (Nbs, Nt, ...)
                return_mbs=True
                    corr_bs: an array of shape (Nbs, Mbs, Nt, ...)
                return_bs_list=True
                    corr_bs, bs_list.shape = (Nbs, Mbs)
        '''

        Ncfg = corr.shape[0]
        if Mbs:
            m_bs = Mbs
        else:
            m_bs = Ncfg

        # seed the random number generator
        rng = get_rng(seed,verbose=verbose) if seed else np.random.default_rng()

        # make BS list: [low, high)
        bs_list = rng.integers(low=0, high=Ncfg, size=[Nbs, m_bs])

        # make BS corrs
        corr_bs = np.zeros(tuple([Nbs, m_bs]) + corr.shape[1:], dtype=corr.dtype)
        for bs in range(Nbs):
            corr_bs[bs] = corr[bs_list[bs]]

        # if return_mbs, return (Nbs, Mbs, Nt, ...) array
        # otherwise, return mean over Mbs axis
        if return_mbs:
            bs_mean   = corr_bs.mean(axis=(0,1))
            d_corr_bs = corr_bs - bs_mean
            corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)
        else:
            corr_bs   = corr_bs.mean(axis=1)
            bs_mean   = corr_bs.mean(axis=0)
            d_corr_bs = corr_bs - bs_mean
            corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)

        if return_bs_list:
            return corr_bs, bs_list
        else:
            return corr_bs


    def bs_prior(Nbs, mean=0., sdev=1., seed=None):
        ''' Generate bootstrap distribution of prior central values
            Args:
                Nbs  : number of values to return
                mean : mean of Gaussian distribution
                sdev : width of Gaussian distribution
                seed : string to seed random number generator
            Return:
                a numpy array of length Nbs of normal(mean, sdev) values
        '''
        # seed the random number generator
        rng = get_rng(seed) if seed else np.random.default_rng()

        return rng.normal(loc=mean, scale=sdev, size=Nbs)


    def block_data(data, bl):
        ''' Generate "blocked" or "binned" data from original data
            Args:
                data : data of shape is (Ncfg, ...)
                bl   : block length in axis=0 units
            Return:
                block data : data of shape (Ncfg//bl, ...)
        '''
        ncfg, nt_gf = data.shape
        if ncfg % bl == 0:
            nb = ncfg // bl
        else:
            nb = ncfg // bl + 1
        corr_bl = np.zeros([nb, nt_gf], dtype=data.dtype)
        for b in range(nb-1):
            corr_bl[b] = data[b*bl:(b+1)*bl].mean(axis=0)
        corr_bl[nb-1] = data[(nb-1)*bl:].mean(axis=0)

        return corr_bl
