import numpy as np
from numpy.random import Generator, SeedSequence, PCG64
import hashlib


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


def get_bs_list(Ndata, Nbs, Mbs=None, seed=None, verbose=False):
    ''' generate bootstrap resampling of correlation function data
        Args:
            - Ndata : Number of data samples (Ncfg for example)
            - Nbs   : the number of bootstrap samples to generate
            - Mbs   : the number of random draws per bootstrap to generate
                      if Mbs != Ncfg, you will have to appropriately rescale
                      the fluctuations by sqrt( Mbs / Ncfg)
            - seed:   a string that will be hashed to seed the random number generator

        Return:
            bs_list : (Nbs, Mbs)
    '''
    if Mbs:
        m_bs = Mbs
    else:
        m_bs = Ndata

    # seed the random number generator
    rng = get_rng(seed,verbose=verbose) if seed else np.random.default_rng()

    # make BS list: [low, high)
    bs_list = rng.integers(low=0, high=Ndata, size=[Nbs, m_bs])

    return bs_list


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


def bs_prior(Nbs, mean=0., sdev=1., seed=None, dist='normal'):
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

    if dist == 'normal':
        return rng.normal(loc=mean, scale=sdev, size=Nbs)
    elif dist == 'lognormal':
        return rng.lognormal(mean=mean, sigma=sdev, size=Nbs)
    else:
        sys.exit('you have not given a known distribution, %s' %dist)

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
