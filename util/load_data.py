import tables as h5
import numpy as np
import gvar as gv


'''
TIME REVERSE
'''
def time_reverse(corr, reverse=True, phase=1, time_axis=1):
    ''' assumes time index is second of array '''
    ''' assumes phase = +- 1 '''
    if reverse:
        if len(corr.shape) > 1:
            cr = phase * np.roll(corr[:,::-1],1,axis=time_axis)
            cr[:,0] = phase * cr[:,0]
        else:
            cr = phase * np.roll(corr[::-1],1)
            cr[0] = phase * cr[0]
    else:
        cr = phase * corr
    return cr

def load_h5(f5_file, corr_dict, return_gv=True):
    corrs = gv.BufferDict()

    # check if f5_file is list
    if not isinstance(f5_file, list):
        f5_files = [f5_file]
    else:
        f5_files = f5_file

    # collect correlators
    for corr in corr_dict:
        dsets     = corr_dict[corr]['dsets']
        weights   = corr_dict[corr]['weights']
        t_reverse = corr_dict[corr]['t_reverse']
        # check if data is in an array or single correlators
        if 'corr_array' not in corr_dict[corr]:
            corr_array = True
        else:
            corr_array = corr_dict[corr]['corr_array']
        if corr_array:
            # get first data
            with h5.open_file(f5_files[0],'r') as f5:
                data = np.zeros_like(f5.get_node('/'+dsets[0]).read())
                for i_d,dset in enumerate(dsets):
                    if 'phase' in corr_dict[corr]:
                        phase = corr_dict[corr]['phase'][i_d]
                    else: phase=1
                    d_tmp = f5.get_node('/'+dset).read()
                    data += weights[i_d] * time_reverse(d_tmp,reverse=t_reverse[i_d], phase=phase)
            # if we have more than 1 data file
            if len(f5_files) > 1:
                for f_i in range(1,len(f5_files)):
                    with h5.open_file(f5_files[f_i],'r') as f5:
                        tmp = np.zeros_like(f5.get_node('/'+dsets[0]).read())
                        for i_d,dset in enumerate(dsets):
                            if 'phase' in corr_dict[corr]:
                                phase = corr_dict[corr]['phase'][i_d]
                            else: phase=1
                            d_tmp = f5.get_node('/'+dset).read()
                            tmp  += weights[i_d] * time_reverse(d_tmp,reverse=t_reverse[i_d], phase=phase)
                        # NOTE - we assume the cfg axis == 0
                        data = np.concatenate((data,tmp),axis=0)
            # if fold
            if corr_dict[corr]['fold']:
                data = 0.5*(data + time_reverse(data))
            # populate into [Ncfg, Nt] arrays
            for i,snk in enumerate(corr_dict[corr]['snks']):
                for j,src in enumerate(corr_dict[corr]['srcs']):
                    if 'normalize' in corr_dict[corr] and corr_dict[corr]['normalize']:
                        corrs[corr+'_'+snk+src] = data[:,:,i,j] / data.mean(axis=0)[0,i,j]
                    else:
                        corrs[corr+'_'+snk+src] = data[:,:,i,j]
                        
        else: # load individual corrs
            for i,snk in enumerate(corr_dict[corr]['snks']):
                for j,src in enumerate(corr_dict[corr]['srcs']):
                    with h5.open_file(f5_files[0],'r') as f5:
                        d_set = dsets[0] % {'SNK':snk, 'SRC':src}
                        data  = np.zeros_like(f5.get_node('/'+d_set).read())
                        for i_d,dset in enumerate(dsets):
                            d_set = dset % {'SNK':snk, 'SRC':src}
                            if 'phase' in corr_dict[corr]:
                                phase = corr_dict[corr]['phase'][i_d]
                            else: phase=1
                            d_tmp = f5.get_node('/'+d_set).read()
                            data += weights[i_d] * time_reverse(d_tmp,reverse=t_reverse[i_d], phase=phase)
                    # if we have more than 1 data file
                    if len(f5_files) > 1:
                        for f_i in range(1,len(f5_files)):
                            with h5.open_file(f5_files[f_i],'r') as f5:
                                d_set = dsets[0] % {'SNK':snk, 'SRC':src}
                                tmp = np.zeros_like(f5.get_node('/'+d_set).read())
                                for i_d,dset in enumerate(dsets):
                                    d_set = dset % {'SNK':snk, 'SRC':src}
                                    if 'phase' in corr_dict[corr]:
                                        phase = corr_dict[corr]['phase'][i_d]
                                    else: phase=1
                                    d_tmp = f5.get_node('/'+d_set).read()
                                    tmp  += weights[i_d] * time_reverse(d_tmp,reverse=t_reverse[i_d], phase=phase)
                                # NOTE - we assume the cfg axis == 0
                                data = np.concatenate((data,tmp),axis=0)
                    # if fold
                    if corr_dict[corr]['fold']:
                        data = 0.5*(data + time_reverse(data))
                    # normalize?
                    if 'normalize' in corr_dict[corr] and corr_dict[corr]['normalize']:
                        corrs[corr+'_'+snk+src] = data / data.mean(axis=0)[0]
                    else:
                        corrs[corr+'_'+snk+src] = data

    # return correlators
    if return_gv:
        corrs_gv = gv.dataset.avg_data(corrs)
        return corrs_gv
    else:
        return corrs
