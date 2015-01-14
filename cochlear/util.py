import os.path

import tables
import numpy as np


def merge_abr_files(filenames, new_filename):
    if os.path.exists(new_filename):
        raise IOError('Output file already exists')
    with tables.open_file(new_filename, 'w') as fh_new:
        trial_logs = []
        waveforms = []
        fs = []
        for filename in filenames:
            with tables.open_file(filename) as fh:
                trial_logs.append(fh.root.trial_log.read())
                waveforms.append(fh.root.waveforms.read())
                fs.append(fh.root.waveforms._v_attrs.fs)
        if len(np.unique(fs)) != 1:
            mesg = 'Cannot merge data collected with different sampling rates'
            raise ValueError(mesg)
        trial_logs = np.concatenate(trial_logs, axis=0)
        waveforms = np.concatenate(waveforms, axis=0)
        fh_new.create_table('/', 'trial_log', trial_logs)
        w_node = fh_new.create_array('/', 'waveforms', waveforms)
        w_node._v_attrs['fs'] = fs[0]

