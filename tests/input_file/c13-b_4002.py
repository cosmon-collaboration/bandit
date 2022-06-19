import gvar as gv
import numpy as np

data_file = 'data/C13/C13-b_4002.ama.h5'

fit_states = ['proton', 'proton_SP', 
              'pion',   'pion_SP', 
              'ext_current', 'ext_current_SP', 
              'local_axial_SP']
#bs_seed = 'a12m220XL'

corr_lst = {
    # PION
    'proton':{
        'dsets':['C13-b_4002.ama.h5/2pt/proton'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,48.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :2,
        'T'        :96,
        't_range'  :np.arange(8,48),
        't_sweep'  :range(3,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    },
}