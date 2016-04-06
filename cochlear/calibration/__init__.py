import logging
log = logging.getLogger(__name__)

import threading

import numpy as np

from neurogen import block_definitions as blocks
from neurogen.calibration import (PointCalibration, InterpCalibration,
                                  CalibrationError, CalibrationTHDError,
                                  CalibrationNFError)
from neurogen import generate_waveform
from neurogen.util import db
from neurogen.calibration.util import (tone_power_conv, csd, psd_freq, thd,
                                       golay_pair, golay_transfer_function)

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
               input_line=ni.DAQmxDefaults.MIC_INPUT, debug=False):

    calibration = InterpCalibration.as_attenuation(vrms=vrms)
    trim_n = int(trim*fs)
    token = blocks.Tone(frequency=frequency, level=0)
    waveform = generate_waveform(token, fs, duration=duration,
                                 calibration=calibration, vrms=vrms)

    daq_kw = {
        'waveform': waveform,
        'repetitions': repetitions,
        'output_line': output_line,
        'input_line': input_line,
        'gain': gain,
        'adc_fs': fs,
        'dac_fs': fs,
        'iti': 0.01
    }

    signal = ni.acquire_waveform(**daq_kw)[:, :, trim_n:-trim_n]

    # Measure the noise floor
    if min_db is not None:
        token = blocks.Silence()
        daq_kw['waveform'] = generate_waveform(token, fs, duration=duration,
                                               calibration=calibration,
                                               vrms=vrms)
        nf_signal = ni.acquire_waveform(**daq_kw)
        nf_signal = nf_signal[:, :, trim_n:-trim_n]
    else:
        nf_signal = np.full_like(signal, np.nan)

    result = _process_tone(frequency, fs, nf_signal, signal, min_db, max_thd)
    if debug:
        return result, signal, nf_signal
    else:
        return result


def tone_spl(frequency, input_calibration, *args, **kwargs):
    rms = tone_power(frequency, *args, **kwargs)[0]
    return input_calibration.get_spl(frequency, rms)


def tone_sens(frequency, input_calibration, gain=-50, vrms=1, *args, **kwargs):
    output_spl = tone_spl(frequency, input_calibration, gain, vrms, *args,
                          **kwargs)
    mesg = 'Output {:.2f}dB SPL at {:.2f}Hz, {:.2f}dB gain, {:.2f}Vrms'
    log.debug(mesg.format(output_spl, frequency, gain, vrms))
    output_sens = _to_sens(output_spl, gain, vrms)
    return output_sens


def tone_calibration(frequency, *args, **kwargs):
    '''
    Single output calibration at a fixed frequency

    Returns
    -------
    sens : dB (V/Pa)
        Sensitivity of output in dB (V/Pa).
    '''
    output_sens = tone_sens(frequency, *args, **kwargs)
    return PointCalibration(frequency, output_sens)


def multitone_calibration(frequencies, *args, **kwargs):
    output_sens = [tone_sens(f, *args, **kwargs) for f in frequencies]
    return PointCalibration(frequencies, output_sens)


def tone_ref_calibration(frequency, gain, input_line=ni.DAQmxDefaults.MIC_INPUT,
                         reference_line=ni.DAQmxDefaults.REF_MIC_INPUT,
                         ref_mic_sens=0.922e-3, *args, **kwargs):

    kwargs['input_line'] = ','.join((input_line, reference_line))
    mic, ref_mic = db(tone_power(frequency, gain, *args, **kwargs))
    sens = mic+db(ref_mic_sens)-ref_mic
    return sens


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
    trim_n = int(trim*fs)

    t1 = blocks.Tone(frequency=f1_frequency, level=0)
    t2 = blocks.Tone(frequency=f2_frequency, level=0)
    w1 = generate_waveform(t1, fs, duration, cal1)[np.newaxis]
    w2 = generate_waveform(t2, fs, duration, cal2)[np.newaxis]
    waveforms = np.concatenate((w1, w2), axis=0)

    daq_kw = {
        'waveform': waveforms,
        'repetitions': repetitions,
        'output_line': ni.DAQmxDefaults.DUAL_SPEAKER_OUTPUT,
        'input_line': ni.DAQmxDefaults.MIC_INPUT,
        'gain': [f1_gain, f2_gain],
        'adc_fs': fs,
        'dac_fs': fs,
        'iti': 0.01
    }

    signal = ni.acquire_waveform(**daq_kw)[:, :, trim_n:-trim_n]

    # Measure the noise floor
    if min_db is not None:
        token = blocks.Silence()
        w1 = generate_waveform(token, fs, duration, cal1)[np.newaxis]
        w2 = generate_waveform(token, fs, duration, cal2)[np.newaxis]
        daq_kw['waveform'] = np.concatenate((w1, w2), axis=0)
        nf_signal = ni.acquire_waveform(**daq_kw)[:, :, trim_n:-trim_n]
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

    mesg = 'Maximum output at {:.1f}Hz is {:.1f}dB SPL'
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


class CalibrationResult(object):
    pass


class ChirpCalibration(object):

    def __init__(self, freq_lb=50, freq_ub=100e3, start_atten=0, end_atten=0,
                 vrms=1, gain=0, repetitions=32, duration=0.1, rise_time=0.001,
                 iti=0.01, fs=200e3, input_range=10,
                 output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 input_line=ni.DAQmxDefaults.MIC_INPUT, callback=None):

        # By using an Attenuation calibration (generated by get_chirp_transform)
        # and setting tone level to 0, a sine wave at the given amplitude (as
        # specified in the settings) will be generated at each frequency as the
        # reference.
        calibration = get_chirp_transform(vrms, start_atten, end_atten)
        ramp = blocks.LinearRamp(name='sweep')
        token = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope')
        token.set_value('sweep.ramp_duration', duration)
        token.set_value('envelope.duration', duration)
        token.set_value('envelope.rise_time', rise_time)
        token.set_value('sweep.start', freq_lb)
        token.set_value('sweep.stop', freq_ub)

        waveform = generate_waveform(token, fs, duration=duration,
                                     calibration=calibration, vrms=vrms)
        daq_kw = {
            'waveform': waveform,
            'repetitions': repetitions,
            'output_line': output_line,
            'input_line': input_line,
            'gain': gain,
            'adc_fs': fs,
            'dac_fs': fs,
            'iti': iti,
            'callback': callback,
            'output_range': 10,
            'input_range': input_range,
        }
        self.iface_acquire = ni.DAQmxAcquireWaveform(**daq_kw)
        self.fs = fs
        self.sig_waveform = waveform
        self.iti = iti

    def acquire(self, join=True):
        self.iface_acquire.start()
        if join:
            self.iface_acquire.join()

    def process(self, fft_window='boxcar', waveform_averages=4,
                input_gains=None):
        # Subtract one from the trim because the DAQmx interface is configured
        # to acquire one sample less than int(waveform_duration+iti).  This
        # allows the card to be reset properly so it can acquire on the next
        # trigger.
        time = np.arange(self.sig_waveform.shape[-1])/self.fs
        mic_waveforms = self.iface_acquire.get_waveforms(remove_iti=True)
        if mic_waveforms.shape[-1] != self.sig_waveform.shape[-1]:
            raise ValueError('shapes do not match')

        if input_gains is not None:
            # Correct for measurement gain settings
            input_gains = np.asarray(input_gains)[..., np.newaxis]
            mic_waveforms = mic_waveforms/input_gains

        mic_frequency = psd_freq(mic_waveforms[0, 0, :], self.fs)
        sig_frequency = psd_freq(self.sig_waveform, self.fs)

        mic_csd = csd(mic_waveforms, self.fs, fft_window, waveform_averages)
        mic_phase = np.unwrap(np.angle(mic_csd)).mean(axis=0)
        mic_psd = np.mean(2*np.abs(mic_csd)/np.sqrt(2.0), axis=0)

        sig_csd = csd(self.sig_waveform, self.fs, fft_window)
        sig_phase = np.unwrap(np.angle(sig_csd))
        sig_psd = 2*np.abs(sig_csd)/np.sqrt(2.0)

        return {
            'fs': self.fs,
            'mic_frequency': mic_frequency,
            'sig_frequency': sig_frequency,
            'mic_psd': mic_psd,
            'sig_psd': sig_psd,
            'mic_phase_raw': mic_phase,
            'mic_phase': mic_phase-sig_phase[np.newaxis],
            'sig_phase': sig_phase,
            'time': time,
            'sig_waveform': self.sig_waveform,
            'mic_waveforms': mic_waveforms,
        }


def chirp_power(waveform_averages=4, fft_window='boxcar', **kwargs):
    c = ChirpCalibration(**kwargs)
    c.acquire()
    return c.process(fft_window, waveform_averages)


class GolayCalibration(object):

    def __init__(self, n=16, vrms=1, gain=0, repetitions=1, iti=0.01,
                 ab_delay=2,  fs=200e3,
                 output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 input_line=ni.DAQmxDefaults.MIC_INPUT, callback=None):

        self.a, self.b = golay_pair(n)
        self.daq_kw = {
            'repetitions': repetitions,
            'output_line': output_line,
            'input_line': input_line,
            'gain': gain,
            'adc_fs': fs,
            'dac_fs': fs,
            'iti': iti,
            'callback': self.poll,
        }
        self.running = None
        self.callback = callback
        self.ab_delay = ab_delay
        self.fs = fs

    def poll(self, epochs_acquired, complete):
        if complete and self.running == 'a':
                self.a_waveforms = \
                    self.iface_acquire.get_waveforms(remove_iti=True)
                self.callback(epochs_acquired, False)
                threading.Timer(self.ab_delay, self.acquire_b).start()
        elif complete and self.running == 'b':
                self.b_waveforms = \
                    self.iface_acquire.get_waveforms(remove_iti=True)
                self.callback(epochs_acquired, True)
        else:
            self.callback(epochs_acquired, False)

    def acquire(self, join=True):
        self.acquire_a()

    def acquire_a(self):
        self.running = 'a'
        self.iface_acquire = \
            ni.DAQmxAcquireWaveform(waveform=self.a, **self.daq_kw)
        self.iface_acquire.start()

    def acquire_b(self):
        self.running = 'b'
        self.iface_acquire = \
            ni.DAQmxAcquireWaveform(waveform=self.b, **self.daq_kw)
        self.iface_acquire.start()

    def process(self, waveform_averages, input_gains=None, discard=1,
                smoothing_window=5):
        result = summarize_golay(self.fs, self.a, self.b,
                                 self.a_waveforms[discard:],
                                 self.b_waveforms[discard:],
                                 waveform_averages,
                                 input_gains)
        mic_waveforms = np.concatenate((self.a_waveforms, self.b_waveforms),
                                       axis=-1)
        sig_waveform = np.concatenate((self.a, self.b), axis=-1)
        sig_csd = csd(sig_waveform, self.fs)
        sig_phase = np.unwrap(np.angle(sig_csd))
        sig_psd = 2*np.abs(sig_csd)/np.sqrt(2.0)
        sig_frequency = psd_freq(sig_waveform, self.fs)

        result.update({
            'mic_waveforms': mic_waveforms,
            'sig_waveform': sig_waveform,
            'sig_psd': sig_psd,
            'sig_phase': sig_phase,
            'sig_frequency': sig_frequency,
        })
        return result


def summarize_golay(fs, a, b, a_response, b_response, waveform_averages=None,
                    input_gains=None):
    n_epochs, n_channels, n_time = a_response.shape
    if waveform_averages is not None:
        new_shape = (waveform_averages, -1, n_channels, n_time)
        a_response = a_response.reshape(new_shape).mean(axis=0)
        b_response = b_response.reshape(new_shape).mean(axis=0)
    if input_gains is not None:
        # Correct for measurement gain settings
        input_gains = np.asarray(input_gains)[..., np.newaxis]
        a_response = a_response/input_gains
        b_response = b_response/input_gains

    time = np.arange(a_response.shape[-1])/fs
    freq, tf_psd, tf_phase = golay_transfer_function(a, b, a_response,
                                                     b_response, fs)
    tf_psd = tf_psd.mean(axis=0)
    tf_phase = tf_phase.mean(axis=0)

    return {
        'fs': fs,
        'a': a,
        'b': b,
        'a_response': a_response,
        'b_response': b_response,
        'time': time,
        'tf_psd': tf_psd,
        'tf_phase': tf_phase,
        'mic_frequency': freq,
    }


def golay_tf(*args, **kwargs):
    c = GolayCalibration(*args, **kwargs)
    c.acquire()
    return c.process()
