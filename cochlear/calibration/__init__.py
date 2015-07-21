import logging
log = logging.getLogger(__name__)

import numpy as np

from neurogen import block_definitions as blocks
from neurogen.calibration import (PointCalibration, InterpCalibration,
                                  CalibrationError, CalibrationTHDError,
                                  CalibrationNFError)
from neurogen.util import db, dbi
from neurogen.calibration.util import (analyze_mic_sens, analyze_tone,
                                       tone_power_conv, psd, psd_freq, thd)

from .. import nidaqmx as ni

################################################################################
# Utility tone calibration functions
################################################################################
thd_err_mesg = 'Total harmonic distortion for {:0.1f}Hz is {:0.2f}%'
nf_err_mesg = 'Power at {:0.1f}Hz of {:0.1f}dB near noise floor of {:0.1f}dB'


def _to_sens(output_spl, output_gain, vrms):
    # Convert SPL to value expected at 0 dB gain and 1 VRMS
    norm_spl = output_spl-output_gain-db(vrms)
    return -norm_spl-db(20e-6)


def _process_tone(frequency, fs, nf_signal, signal, min_db, max_thd):
    rms = tone_power_conv(signal, fs, frequency, 'flattop')
    rms_average = np.mean(rms, axis=0)

    if max_thd is not None:
        measured_thd = thd(signal, fs, frequency, 3, 'flattop')
        measured_thd_average = np.mean(measured_thd, axis=0)
    else:
        measured_thd_average = np.full_like(rms_average, np.nan)

    if min_db is not None:
        nf_rms = tone_power_conv(nf_signal, fs, frequency, 'flattop')
        nf_rms_average = np.mean(nf_rms, axis=0)
    else:
        nf_rms_average = np.full_like(rms_average, np.nan)

    for n, s, t in zip(nf_rms_average, rms_average, measured_thd_average):
        mesg = 'Noise floor {:.1f}dB, signal {:.1f}dB, THD {:.2f}%'
        log.debug(mesg.format(db(n), db(s), t*100))
        _check_calibration(frequency, s, n, min_db, t, max_thd)
    return rms_average


def _check_calibration(frequency, rms, nf_rms, min_db, thd, max_thd):
    if min_db is not None and (db(rms, nf_rms) < min_db):
        m = nf_err_mesg.format(frequency, db(rms), db(nf_rms))
        raise CalibrationNFError(m)
    if max_thd is not None and (thd > max_thd):
        m = thd_err_mesg.format(frequency, thd*100)
        raise CalibrationTHDError(m)


def tone_power(frequency, gain=0, vrms=1, repetitions=1, fs=200e3, max_thd=0.1,
               min_db=10, duration=0.1, trim=0.01,
               output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
               input_line=ni.DAQmxDefaults.MIC_INPUT):

    calibration = InterpCalibration.as_attenuation(vrms=vrms)
    c = ni.DAQmxChannel(calibration=calibration)
    trim_n = int(trim*fs)

    daq_kw = {
        'channels': [c],
        'repetitions': repetitions,
        'output_line': output_line,
        'input_line': input_line,
        'gain': gain,
        'adc_fs': fs,
        'dac_fs': fs,
        'duration': duration,
        'iti': 0.01
    }

    # Measure the actual output
    c.token = blocks.Tone(frequency=frequency, level=0)
    signal = ni.acquire(**daq_kw)[:, :, trim_n:-trim_n]

    # Measure the noise floor
    if min_db is not None:
        c.token = blocks.Silence()
        nf_signal = ni.acquire(**daq_kw)
        nf_signal = nf_signal[:, :, trim_n:-trim_n]
    else:
        nf_signal = np.full_like(signal, np.nan)

    return _process_tone(frequency, fs, nf_signal, signal, min_db, max_thd)


def tone_spl(frequency, input_calibration, *args, **kwargs):
    rms = tone_power(frequency, *args, **kwargs)[0]
    return input_calibration.get_spl(frequency, rms)


def tone_calibration(frequency, input_calibration, gain=-50, vrms=1, *args,
                     **kwargs):
    '''
    Single output calibration at a fixed frequency

    Returns
    -------
    sens : dB (V/Pa)
        Sensitivity of output in dB (V/Pa).
    '''
    output_spl = tone_spl(frequency, input_calibration, gain, vrms, *args,
                          **kwargs)
    mesg = 'Output {:.2f}dB SPL at {:.2f}Hz, {:.2f}dB gain, {:.2f}Vrms'
    log.debug(mesg.format(output_spl, frequency, gain, vrms))
    output_sens = _to_sens(output_spl, gain, vrms)
    return PointCalibration(frequency, output_sens)


def tone_calibration_search(frequency, input_calibration, gains, vrms=1,
                            callback=None, *args, **kwargs):
    for gain in gains:
        try:
            if callback is not None:
                callback(gain)
            return tone_calibration(frequency, input_calibration, gain, vrms,
                                    *args, **kwargs)
        except CalibrationError:
            pass
    else:
        raise SystemError('Could not calibrate speaker')


def two_tone_power(f1_frequency, f2_frequency, f1_gain=-50.0, f2_gain=-50.0,
                   f1_vrms=1, f2_vrms=1, repetitions=1, fs=200e3,
                   max_thd=0.01, min_db=10, duration=0.1, trim=0.01):
    '''
    Dual output calibration with each output at a different frequency

    .. note::
        If one frequency is a harmonic of the other, the calibration will fail
        due to the THD measure.  This function is typically most useful for
        calibration of the f1 and f2 DPOAE frequencies (which are not harmonics
        of each other).
    '''
    cal1 = InterpCalibration.as_attenuation(vrms=f1_vrms)
    cal2 = InterpCalibration.as_attenuation(vrms=f2_vrms)
    c1 = ni.DAQmxChannel(calibration=cal1)
    c2 = ni.DAQmxChannel(calibration=cal2)
    trim_n = int(trim*fs)

    daq_kw = {
        'channels': [c1, c2],
        'repetitions': repetitions,
        'output_line': ni.DAQmxDefaults.DUAL_SPEAKER_OUTPUT,
        'input_line': ni.DAQmxDefaults.MIC_INPUT,
        'gain': (f2_gain, f1_gain),
        'adc_fs': fs,
        'dac_fs': fs,
        'duration': duration,
        'iti': 0.01
    }

    # Measure the actual output
    c1.token = blocks.Tone(frequency=f1_frequency, level=0)
    c2.token = blocks.Tone(frequency=f2_frequency, level=0)
    signal = ni.acquire(**daq_kw)[:, :, trim_n:-trim_n]

    # Measure the noise floor
    if min_db is not None:
        print 'measuring silence'
        c1.token = blocks.Silence()
        c2.token = blocks.Silence()
        nf_signal = ni.acquire(**daq_kw)[:, :, trim_n:-trim_n]
    else:
        nf_signal = np.full_like(signal, np.nan)

    f1 = _process_tone(f1_frequency, fs, nf_signal, signal, min_db, max_thd)
    f2 = _process_tone(f2_frequency, fs, nf_signal, signal, min_db, max_thd)
    return f1, f2


def two_tone_spl(f1_frequency, f2_frequency, input_calibration, *args,
                 **kwargs):
    '''
    Dual measurement of output SPL

    .. note::
        If one frequency is a harmonic of the other, the calibration will fail
        due to the THD measure.  This function is typically most useful for
        calibration of the f1 and f2 DPOAE frequencies (which are not harmonics
        of each other).
    '''
    f1_rms, f2_rms = two_tone_power(f1_frequency, f2_frequency, *args, **kwargs)
    f1_spl = input_calibration.get_spl(f1_frequency, f1_rms)[0]
    f2_spl = input_calibration.get_spl(f2_frequency, f2_rms)[0]
    return f1_spl, f2_spl


def two_tone_calibration(f1_frequency, f2_frequency, input_calibration,
                         f1_gain=-50, f2_gain=-50, f1_vrms=1, f2_vrms=1, *args,
                         **kwargs):
    '''
    Dual output calibration with each output at a different frequency

    .. note::
        If one frequency is a harmonic of the other, the calibration will fail
        due to the THD measure.  This function is typically most useful for
        calibration of the f1 and f2 DPOAE frequencies (which are not harmonics
        of each other).
    '''
    f1_spl, f2_spl = two_tone_spl(f1_frequency, f2_frequency, input_calibration,
                                  f1_gain, f2_gain, f1_vrms, f2_vrms, *args,
                                  **kwargs)
    mesg = '{} output {:.2f}dB SPL at {:.2f}Hz, {:.2f}dB gain, {:.2f}Vrms'
    log.debug(mesg.format('Primary', f1_spl, f1_frequency, f1_gain, f1_vrms))
    log.debug(mesg.format('Secondary', f2_spl, f2_frequency, f2_gain, f2_vrms))
    f1_sens = _to_sens(f1_spl, f1_gain, f1_vrms)
    f2_sens = _to_sens(f2_spl, f2_gain, f2_vrms)
    return PointCalibration(f1_frequency, f1_sens), \
        PointCalibration(f2_frequency, f2_sens)


def ceiling_spl(frequency, max_spl=80, initial_gain=-40, vrms=1, spl_step=5,
                gain_step=5, **cal_kw):
    '''
    Return maximum SPL at given frequency without distortion of output
    '''
    step_size = gain_step
    last_step = None
    ceiling_spl = None
    output_gain = initial_gain

    # Determine the starting output gain to achieve the maximum output.  At this
    # point we are going to ignore THD; however, we need to make sure we are
    # measuring above the noise floor.
    initial_cal_kw = cal_kw.copy()
    initial_cal_kw['max_thd'] = np.inf
    while True:
        try:
            spl = tone_calibration(frequency, output_gain, **initial_cal_kw)
            output_gain += max_spl-spl
            output_gain = np.round(output_gain/0.5)*0.5
            break
        except CalibrationNFError:
            output_gain += step_size

    while True:
        try:
            spl = tone_calibration(frequency, output_gain, **cal_kw)
            if np.abs(spl-max_spl) < 1:
                ceiling_spl = spl
                break
            else:
                output_gain += max_spl-spl
                output_gain = np.round(output_gain/0.5)*0.5
                last_step = max_spl-spl
        except CalibrationNFError:
            # We have descended too close to the noise floor
            if last_step is not None and last_step < 0:
                step_size = int(step_size/2)
            output_gain += step_size
            last_step = step_size
        except CalibrationTHDError:
            max_spl -= spl_step
            if last_step is not None and last_step > 0:
                step_size = int(step_size/2)
            output_gain -= step_size
            last_step = -step_size
        if step_size <= 1:
            break

    if ceiling_spl is None:
        raise CalibrationError('Could not determine maximum SPL')

    mesg ='Maximum output at {:.1f}Hz is {:.1f}dB SPL'
    log.debug(mesg.format(frequency, ceiling_spl))
    return ceiling_spl


def mic_sens(frequency, ref_input, exp_input, ref_calibration, *args, **kwargs):
    '''
    Compute sensitivity of experiment microphone (e.g. probe tube microphone)
    based on the reference microphone and sensitivity for the reference
    microphone.

    Parameters
    ----------
    frequency : float (Hz)
        Frequency to calibrate at
    ref_input : str
        niDAQmx input channel for reference microphone
    exp_input : str
        niDAQmx input channel for experiment microphone
    '''
    mic_input = ','.join((ref_input, exp_input))
    ref_power, exp_power = tone_power(frequency, *args, input_line=mic_input,
                                      **kwargs)
    ref_sens = ref_calibration.get_sens()
    return db(exp_power)+db(ref_power)-db(ref_sens)


################################################################################
# Utility chirp calibration functions
################################################################################
def get_chirp_transform(vrms, start_atten=6, end_atten=-6, start_frequency=0,
                        end_frequency=100e3):
    frequencies = [start_frequency, end_frequency]
    magnitude = [start_atten, end_atten]
    return InterpCalibration.from_single_vrms(frequencies, magnitude, vrms)


class ChirpCalibration(object):

    def __init__(self, freq_lb, freq_ub, start_atten=0, end_atten=0, vrms=1,
                 gain=0, repetitions=1, duration=0.1, rise_time=0.1, iti=0.01,
                 fs=200e3, output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 input_line=ni.DAQmxDefaults.MIC_INPUT, callback=None):

        for k, v in locals().items():
            setattr(self, k, v)

        calibration = get_chirp_transform(vrms, start_atten, end_atten)

        ramp = blocks.LinearRamp(name='sweep')
        channel = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope') >> \
            ni.DAQmxChannel(calibration=calibration)

        channel.set_value('sweep.ramp_duration', duration)
        channel.set_value('envelope.duration', duration)
        channel.set_value('envelope.rise_time', rise_time)
        channel.set_value('sweep.start', freq_lb)
        channel.set_value('sweep.stop', freq_ub)

        daq_kw = {
            'channels': [channel],
            'repetitions': repetitions,
            'output_line': output_line,
            'input_line': input_line,
            'gain': gain,
            'adc_fs': fs,
            'dac_fs': fs,
            'duration': duration,
            'iti': iti,
            'callback': callback,
        }
        self.iface_acquire = ni.DAQmxAcquire(**daq_kw)

    def acquire(self):
        self.iface_acquire.start()
        self.iface_acquire.join()

    def process(self, fft_window='boxcar'):
        return self.iface_acquire.waveform_buffer


def chirp_power(*args, **kwargs):
    c = ChirpCalibration(*args, **kwargs)
    c.acquire()
    return c.process()


class GolayCalibration(object):

    def __init__(self, n, vrms=1, gain=0, repetitions=1, iti=0.01, fs=200e3,
                 output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 input_line=ni.DAQmxDefaults.MIC_INPUT, callback=None):

        for k, v in locals().items():
            setattr(self, k, v)

        a, b = golay_pair(n)
        daq_kw = {
            'channels': [channel],
            'repetitions': repetitions,
            'output_line': output_line,
            'input_line': input_line,
            'gain': gain,
            'adc_fs': fs,
            'dac_fs': fs,
            'duration': duration,
            'iti': iti,
            'callback': callback,
        }
        self.iface_acquire = ni.DAQmxAcquire(**daq_kw)

    def acquire(self):
        self.iface_acquire.start()
        self.iface_acquire.join()

    def process(self, fft_window='boxcar'):
        return self.iface_acquire.waveform_buffer