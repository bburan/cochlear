import os.path

import tables
import numpy as np

from neurogen.calibration import InterpCalibration


def truncate_file(filename, new_filename, new_size):
    if os.path.exists(new_filename):
        raise IOError('Output file already exists')
    with tables.open_file(new_filename, 'w') as fh_new:
        with tables.open_file(filename, 'r') as fh:
            fh_new.copyNode(fh.root.trial_log, fh_new.root)
            fh_new.copyNode(fh.root.waveforms, fh_new.root)
        fh_new.root.trial_log.truncate(new_size)
        fh_new.root.waveforms.truncate(new_size)


def merge_files(filenames, new_filename):
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


def get_chirp_transform(vrms, start_atten=6, end_atten=-6):
    calibration_data = np.array([
        (0, start_atten),
        (100e3, end_atten),
    ])
    frequencies = calibration_data[:, 0]
    magnitude = calibration_data[:, 1]
    return InterpCalibration.from_single_vrms(frequencies, magnitude, vrms)
