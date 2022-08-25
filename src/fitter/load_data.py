import tables as h5
import numpy as np
import gvar as gv
import sys


'''
TIME REVERSE
'''


def time_reverse(corr, reverse=True, phase=1, time_axis=1):
    ''' assumes time index is second of array
        assumes phase = +- 1
    '''
    if reverse:
        if len(corr.shape) > 1:
            cr = phase * np.roll(corr[:, ::-1], 1, axis=time_axis)
            cr[:, 0] = phase * cr[:, 0]
        else:
            cr = phase * np.roll(corr[::-1], 1)
            cr[0] = phase * cr[0]
    else:
        cr = phase * corr
    return cr


def load_h5(f5_file, corr_dict, return_gv=True, rw=None, bl=1, uncorr_corrs=False, uncorr_all=False, verbose=True):
    corrs = gv.BufferDict()

    # check if f5_file is list
    if not isinstance(f5_file, list):
        f5_files = [f5_file]
    else:
        f5_files = f5_file
    # check for re-weighting
    if rw:
        rw_file, rw_path = rw
        if not isinstance(rw_file, list):
            rw_files = [rw_file]
        else:
            rw_files = rw_files
        if len(rw_files) != len(f5_files):
            sys.exit(
                'You must supply the same number of re-weighting files as data files')
        with h5.open_file(rw_files[0], 'r') as rw5:
            reweight = rw5.get_node('/'+rw_path).read()
        for f_i in range(1, len(f5_files)):
            with h5.open_file(rw_files[f_i], 'r') as rw5:
                reweight = np.concatenate(
                    (reweight, rw5.get_node('/'+rw_path).read()), axis=0)
        # normalize rw factors
        reweight = reweight / reweight.sum()

    # collect correlators
    for corr in corr_dict:
        weights   = corr_dict[corr]['weights']
        t_reverse = corr_dict[corr]['t_reverse']
        # check if data is in an array or single correlators
        if 'corr_array' not in corr_dict[corr]:
            corr_array = True
        else:
            corr_array = corr_dict[corr]['corr_array']
        if corr_array:
            # get first data
            with h5.open_file(f5_files[0], 'r') as f5:
                dsets = corr_dict[corr]['dsets']
                data = np.zeros_like(f5.get_node('/'+dsets[0]).read())
                for i_d, dset in enumerate(dsets):
                    if 'phase' in corr_dict[corr]:
                        phase = corr_dict[corr]['phase'][i_d]
                    else:
                        phase = 1
                    d_tmp = f5.get_node('/'+dset).read()
                    data += weights[i_d] * \
                        time_reverse(
                            d_tmp, reverse=t_reverse[i_d], phase=phase)

            # if we have more than 1 data file
            if len(f5_files) > 1:
                for f_i in range(1, len(f5_files)):
                    with h5.open_file(f5_files[f_i], 'r') as f5:
                        tmp = np.zeros_like(f5.get_node('/'+dsets[0]).read())
                        for i_d, dset in enumerate(dsets):
                            if 'phase' in corr_dict[corr]:
                                phase = corr_dict[corr]['phase'][i_d]
                            else:
                                phase = 1
                            d_tmp = f5.get_node('/'+dset).read()
                            tmp += weights[i_d] * time_reverse(
                                d_tmp, reverse=t_reverse[i_d], phase=phase)
                        # NOTE - we assume the cfg axis == 0
                        data = np.concatenate((data, tmp), axis=0)

            # if fold
            if corr_dict[corr]['fold']:
                data = 0.5*(data + time_reverse(data))
            # populate into [Ncfg, Nt] arrays
            for i, snk in enumerate(corr_dict[corr]['snks']):
                for j, src in enumerate(corr_dict[corr]['srcs']):
                    if 'normalize' in corr_dict[corr] and corr_dict[corr]['normalize']:
                        corrs[corr+'_'+snk+src] = data[:, :, i, j] / \
                            data.mean(axis=0)[0, i, j]
                    else:
                        corrs[corr+'_'+snk+src] = data[:, :, i, j]

        else:  # load individual corrs
            if corr_dict[corr]['type'] == 'mres':
                with h5.open_file(f5_files[0], 'r') as f5:
                    data_MP = f5.get_node('/'+corr_dict[corr]['dset_MP'][0]).read()
                    data_PP = f5.get_node('/'+corr_dict[corr]['dset_PP'][0]).read()
                    # stack the data so it can be treated like other dsets
                    data = np.stack((data_MP,data_PP),axis=-1)
                    if len(f5_files) > 1:
                        for f_i in range(1, len(f5_files)):
                            with h5.open_file(f5_files[f_i], 'r') as f5:
                                data_MP = f5.get_node('/'+corr_dict[corr]['dset_MP'][0]).read()
                                data_PP = f5.get_node('/'+corr_dict[corr]['dset_PP'][0]).read()
                                tmp     = np.stack((data_MP,data_PP),axis=-1)
                                data    = np.concatenate((data, tmp), axis=0)
                # if fold
                if corr_dict[corr]['fold']:
                    data = 0.5*(data + time_reverse(data))
                # add MP and PP data
                corrs[corr+'_MP'] = data[...,0]
                corrs[corr+'_PP'] = data[...,1]

            else:
                for i, snk in enumerate(corr_dict[corr]['snks']):
                    for j, src in enumerate(corr_dict[corr]['srcs']):
                        with h5.open_file(f5_files[0], 'r') as f5:
                            d_set = dsets[0] % {'SNK': snk, 'SRC': src}
                            data = np.zeros_like(f5.get_node('/'+d_set).read())
                            for i_d, dset in enumerate(dsets):
                                d_set = dset % {'SNK': snk, 'SRC': src}
                                if 'phase' in corr_dict[corr]:
                                    phase = corr_dict[corr]['phase'][i_d]
                                else:
                                    phase = 1
                                d_tmp = f5.get_node('/'+d_set).read()
                                data += weights[i_d] * time_reverse(
                                    d_tmp, reverse=t_reverse[i_d], phase=phase)

                        # if we have more than 1 data file
                        if len(f5_files) > 1:
                            for f_i in range(1, len(f5_files)):
                                with h5.open_file(f5_files[f_i], 'r') as f5:
                                    d_set = dsets[0] % {'SNK': snk, 'SRC': src}
                                    tmp = np.zeros_like(
                                        f5.get_node('/'+d_set).read())
                                    for i_d, dset in enumerate(dsets):
                                        d_set = dset % {'SNK': snk, 'SRC': src}
                                        if 'phase' in corr_dict[corr]:
                                            phase = corr_dict[corr]['phase'][i_d]
                                        else:
                                            phase = 1
                                        d_tmp = f5.get_node('/'+d_set).read()
                                        tmp += weights[i_d] * time_reverse(
                                            d_tmp, reverse=t_reverse[i_d], phase=phase)
                                    # NOTE - we assume the cfg axis == 0
                                    data = np.concatenate((data, tmp), axis=0)
                        # if fold
                        if corr_dict[corr]['fold']:
                            data = 0.5*(data + time_reverse(data))
                        # normalize?
                        if 'normalize' in corr_dict[corr] and corr_dict[corr]['normalize']:
                            #print('normalizing %s %s %s' %(corr,snk,src))
                            corrs[corr+'_'+snk+src] = data / data.mean(axis=0)[0]
                        else:
                            #print('not normalizing %s %s %s' %(corr,snk,src))
                            corrs[corr+'_'+snk+src] = data

    # re-weight?
    if rw:
        for k in corrs:
            corrs[k] = corrs[k] * reweight[:, None]

    # block/bin data
    if bl != 1:
        print('blocking data in units of saved configs: block length = %d' % bl)
        corrs_bl = {}
        for corr in corrs:
            corrs_bl[corr] = block_data(corrs[corr], bl)
        corrs = corrs_bl

    # return correlators
    if verbose:
        for corr in corrs:
            print(corr, corrs[corr].shape)
    if return_gv:
        if uncorr_corrs or uncorr_all:
            corrs_gv = {}
            if uncorr_all:
                for k in corrs:
                    corrs_gv[k] = gv.dataset.avg_data(corrs[k])
            else:
                for corr in corr_dict:
                    corrs_corr = {k: v for k, v in corrs.items() if corr in k}
                    tmp_gv = gv.dataset.avg_data(corrs_corr)
                    for k in tmp_gv:
                        corrs_gv[k] = tmp_gv[k]
        else:
            corrs_gv = gv.dataset.avg_data(corrs)
        return corrs_gv
    else:
        return corrs


def block_data(data, bl):
    ''' data shape is [Ncfg, others]
        bl = block length in configs
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

def svd_processor(data):
    d = gv.dataset.avg_data(data)
    if any(['mres' in k for k in d]):
        d2 = gv.BufferDict({k:v for k,v in d.items() if 'mres' not in k})
        mres = list(set([k.split('_')[0] for k in d if 'mres' in k]))
        for k in mres:
            d2[k] = d[k+'_MP'] / d[k+'_PP']
        d = d2
    return d

def svd_diagnose(data, data_cfg, x_params, nbs, svdcut=None):
    data_chop = dict()
    for d in data:
        if d in x_params and 'mres' not in d:
            data_chop[d] = data_cfg[d][:,x_params[d]['t_range']]
        if 'mres' in d and len(d.split('_')) > 1:
            data_chop[d] = data_cfg[d][:,x_params[d.split('_')[0]]['t_range']]

    svd_test = gv.dataset.svd_diagnosis(data_chop, nbstrap=nbs,
                                        process_dataset=svd_processor)
    svd_cut = svd_test.svdcut
    if svdcut is not None:
        print('  svd_diagnose.svdcut = %.2e' %svd_test.svdcut)
        print('          args.svdcut = %.2e' %svdcut)
        use_svd = input('   use specified svdcut instead of that from svd_diagnosis? [y/n]\n')
        if use_svd in ['y','Y','yes']:
            svd_cut = svdcut

    return svd_test, svd_cut
