import tables as h5
import numpy as np
import gvar as gv


'''
TIME REVERSE
'''
def time_reverse(corr,phase=1,time_axis=1):
    ''' assumes time index is second of array '''
    ''' assumes phase = +- 1 '''
    if len(corr.shape) > 1:
        cr = phase * np.roll(corr[:,::-1],1,axis=time_axis)
        cr[:,0] = phase * cr[:,0]
    else:
        cr = phase * np.roll(corr[::-1],1)
        cr[0] = phase * cr[0]
    return cr

def load_h5(f5_file, corr_dict, return_gv=True):
    corrs = gv.BufferDict()
    with h5.open_file(f5_file,'r') as f5:
        for corr in corr_dict:
            dsets     = corr_dict[corr]['dsets']
            weights   = corr_dict[corr]['weights']
            t_reverse = corr_dict[corr]['t_reverse']
            # get data
            data = np.zeros_like(f5.get_node('/'+dsets[0]).read())
            for i_d,dset in enumerate(dsets):
                if t_reverse[i_d]:
                    data += weights[i_d] * time_reverse(f5.get_node('/'+dsets[i_d]).read(), phase=corr_dict[corr]['phase'][i_d])
                else:
                    data += weights[i_d] * f5.get_node('/'+dsets[i_d]).read()
            if corr_dict[corr]['fold']:
                data = 0.5*(data + time_reverse(data))
            for i,snk in enumerate(corr_dict[corr]['snks']):
                for j,src in enumerate(corr_dict[corr]['srcs']):
                    corrs[corr+'_'+snk+src] = data[:,:,i,j]
    if return_gv:
        corrs_gv = gv.dataset.avg_data(corrs)
        return corrs_gv
    else:
        return corrs
