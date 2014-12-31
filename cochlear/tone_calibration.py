from __future__ import division

import logging
log = logging.getLogger(__name__)

import time

import numpy as np
import matplotlib as mp
import tables
import pyfftw

from traits.api import (HasTraits, Float, Int, Property, Enum, Bool, Instance,
                        Str, List, Array, cached_property)
from traitsui.api import (Item, VGroup, View, Include, ToolBar, Action,
                          Controller, HSplit, HGroup, ListStrEditor, Tabbed,
                          ListEditor)
from pyface.api import ImageResource
from chaco.api import (ArrayPlotData, Plot, DataRange1D, LinearMapper,
                       LogMapper, OverlayPlotContainer, HPlotContainer,
                       VPlotContainer, DataLabel)
from chaco.tools.api import PanTool, ZoomTool, DataLabelTool
from enable.api import Component, ComponentEditor

from cochlear import nidaqmx as ni
from experiment import icon_dir
from experiment.util import get_save_file

from neurogen import block_definitions as blocks
from neurogen.calibration import (PointCalibration, InterpCalibration,
                                  CalibrationError, CalibrationTHDError,
                                  CalibrationNFError)
from neurogen.util import db, dbi
from neurogen.calibration.util import (analyze_mic_sens, analyze_tone,
                                       tone_power_conv, psd, psd_freq, thd)

import settings
from cochlear import calibration_standard as reference_calibration

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
    nf_rms = tone_power_conv(nf_signal, fs, frequency, 'flattop')
    nf_rms_average = np.mean(nf_rms, axis=0)
    measured_thd = thd(signal, fs, frequency, 3, 'flattop')
    measured_thd_average = np.mean(measured_thd, axis=0)
    rms = tone_power_conv(signal, fs, frequency, 'flattop')
    rms_average = np.mean(rms, axis=0)

    for n, s, t in zip(nf_rms_average, rms_average, measured_thd_average):
        mesg = 'Noise floor {:.1f}dB, signal {:.1f}dB, THD {:.2f}%'
        log.debug(mesg.format(db(n), db(s), t*100))
        _check_calibration(frequency, s, n, min_db, t, max_thd)
    return rms_average


def _check_calibration(frequency, rms, nf_rms, min_db, thd, max_thd):
    if db(rms, nf_rms) < min_db:
        m = nf_err_mesg.format(frequency, db(rms), db(nf_rms))
        raise CalibrationNFError(m)
    if thd > max_thd:
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

    # Measure the noise floor
    c.token = blocks.Silence()
    nf_signal = ni.acquire(**daq_kw)
    nf_signal = nf_signal[:, :, trim_n:-trim_n]

    # Measure the actual output
    c.token = blocks.Tone(frequency=frequency, level=0)
    signal = ni.acquire(**daq_kw)[:, :, trim_n:-trim_n]
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


def tone_calibration_search(frequency, input_calibration, gains, vrms=1, *args,
                            **kwargs):
    for gain in gains:
        try:
            return tone_calibration(frequency, input_calibration, gain, vrms,
                                    *args, **kwargs)
        except CalibrationError:
            pass

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

    # Measure the noise floor
    c1.token = blocks.Silence()
    c2.token = blocks.Silence()
    nf_signal = ni.acquire(**daq_kw)[:, 0, trim_n:-trim_n]

    # Measure the actual output
    c1.token = blocks.Tone(frequency=f1_frequency, level=0)
    c2.token = blocks.Tone(frequency=f2_frequency, level=0)
    signal = ni.acquire(**daq_kw)[:, 0, trim_n:-trim_n]

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
    f1_spl = input_calibration.get_spl(f1_frequency, f1_rms)
    f2_spl = input_calibration.get_spl(f1_frequency, f2_rms)
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
# Calibration GUI
################################################################################

class MicToneCalibrationResult(HasTraits):

    frequency = Float(save=True)
    time = Array(save=True)
    freq_psd = Array(save=True)
    ref_mic_rms = Float(save=True, label='Ref. mic. RMS')
    ref_mic_psd = Array(save=True)
    ref_thd = Float(save=True)
    ref_mic_waveform = Array(save=True)
    exp_mic_rms = Float(save=True, label='Exp. mic. RMS')
    exp_mic_psd = Array(save=True)
    exp_thd = Float(save=True)
    exp_mic_waveform = Array(save=True)
    output_spl = Float(save=True, label='Output (dB SPL)')
    norm_output_spl = Float(save=True, label='Norm. output (dB SPL)')
    exp_mic_sens = Float(save=True)
    waveforms = Array(save=True)
    output_gain = Float(save=True)

    harmonics = List()

    waveform_plots = Instance(Component)
    spectrum_plots = Instance(Component)
    harmonic_plots = Instance(Component)

    traits_view = View(
        VGroup(
            HGroup(
                VGroup(
                    'exp_mic_rms',
                    'ref_mic_rms',
                    'output_spl',
                    'norm_output_spl',
                ),
                VGroup(
                    Item('exp_thd', label='THD (frac)'),
                    Item('ref_thd', label='THD (frac)'),
                    Item('output_gain', label='Gain (dB)')
                ),
            ),
            VGroup(
                Item('waveform_plots', editor=ComponentEditor(size=(600, 250))),
                Item('spectrum_plots', editor=ComponentEditor(size=(600, 250))),
                Item('harmonic_plots', editor=ComponentEditor(size=(600, 250))),
                show_labels=False,
            ),
            style='readonly',
        )
    )


class BaseToneCalibrationController(Controller):

    calibration_accepted = Bool(False)

    # Is handler currently acquiring data?
    running = Bool(False)

    # Has data been successfully acquired (and ready for save)?
    acquired = Bool(False)

    epochs_acquired = Int(0)
    iface_daq = Instance('cochlear.nidaqmx.DAQmxAcquire')
    model = Instance('BaseToneCalibration')
    current_frequency = Float(label='Current frequency (Hz)')
    frequencies = List

    gain_step = Float(10)
    current_gain = Float(-20)
    max_gain = Float(31.5)
    min_gain = Float(-96.5)
    current_max_gain = Float(31.5, label='Current max. gain (dB)')

    waveform_buffer = Instance('numpy.ndarray')
    fft_buffer = Instance('numpy.ndarray')
    fftw = Instance('pyfftw.FFTW')

    def setup(self):
        self.running = True
        self.acquired = False
        self.epochs_acquired = 0

        calibration = InterpCalibration.as_attenuation(vrms=self.model.vrms)
        token = blocks.Tone(frequency=self.current_frequency, level=0)
        channel = ni.DAQmxChannel(calibration=calibration, token=token)
        epochs = self.model.waveform_averages*self.model.fft_averages

        self.iface_daq = ni.DAQmxAcquire([channel],
                                         epochs,
                                         output_line=self.model.output,
                                         input_line=self.model.inputs,
                                         gain=self.current_gain,
                                         dac_fs=self.model.fs,
                                         adc_fs=self.model.fs,
                                         duration=self.model.duration,
                                         callback=self.update_status,
                                         )

    def next_frequency(self):
        if self.frequencies:
            self.current_gain = self.max_gain
            self.current_max_gain = self.max_gain
            self.current_frequency = self.frequencies.pop(0)
            self.setup()
            self.iface_daq.start()
        else:
            self.acquired = True
            self.running = False

    def update_status(self, acquired, done):
        self.epochs_acquired = acquired
        if done:
            waveforms = self.iface_daq.waveform_buffer
            shape = list(waveforms.shape)
            shape.insert(0, self.model.waveform_averages)
            shape[1] = self.model.fft_averages
            waveforms = waveforms.reshape(shape).mean(axis=0)
            results = self.analyze(waveforms)
            next_gain = self.next_gain(results)
            if next_gain == self.current_gain:
                self.update_plots(waveforms, results)
                self.next_frequency()
            elif next_gain >= self.current_max_gain:
                self.update_plots(waveforms, results)
                self.next_frequency()
            elif next_gain <= self.min_gain:
                self.update_plots(waveforms, results)
                self.next_frequency()
            else:
                self.current_gain = next_gain
                self.setup()
                self.iface_daq.start()

    def start(self, info):
        self.model = info.object
        self.model.tone_data = []
        self.frequencies = self.model.frequency.tolist()
        self.next_frequency()

    def stop(self, info=None):
        self.iface_daq.stop()

    def save(self, info=None):
        def save_traits(fh, obj, node):
            for trait, value in obj.trait_get(save=True).items():
                if not isinstance(value, basestring) and np.iterable(value):
                    fh.create_array(node, trait, value)
                else:
                    fh.set_node_attr(node, trait, value)

        filename = get_save_file(settings.CALIBRATION_DIR,
                                 'Microphone calibration with tone|*.mic')
        if filename is None:
            return
        with tables.open_file(filename, 'w') as fh:
            save_traits(fh, self.model, fh.root)
            for td in self.model.tone_data:
                node_name = 'frequency_{}'.format(td.frequency)
                td_node = fh.create_group(fh.root, node_name)
                save_traits(fh, td, td_node)

    def accept_calibration(self, info):
        self.calibration_accepted = True
        info.ui.dispose()

    def cancel_calibration(self, info):
        self.calibration_accepted = False
        info.ui.dispose()

    def run_reference_calibration(self, info):
        reference_calibration.launch_gui(parent=info.ui.control,
                                         kind='livemodal')


class BaseToneCalibration(HasTraits):

    # Calibration settings
    fs = Float(400e3, label='AO/AI sampling rate (Hz)', save=True)

    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', save=True)

    output = Enum(('/Dev1/ao0', '/Dev1/ao1'), label='Output (channel)', save=True)
    input_options = ['/Dev1/ai{}'.format(i) for i in range(4)]
    exp_input = Enum('/Dev1/ai1', input_options, label='Exp. mic. (channel)', save=True)
    inputs = Property(depends_on='exp_input, ref_input', save=True)
    output_gain = Float(31.5, label='Output gain (dB)', save=True)

    waveform_averages = Int(1, label='Number of tones per FFT', save=True)
    fft_averages = Int(1, label='Number of FFTs', save=True)
    iti = Float(0.001, label='Inter-tone interval', save=True)
    trim = Float(0.001, label='Trim onset (sec)', save=True)

    start_octave = Float(-2, label='Start octave', save=True)
    start_frequency = Property(depends_on='start_octave', label='End octave', save=True)
    end_octave = Float(6, save=True)
    end_frequency = Property(depends_on='end_octave', save=True)
    octave_spacing = Float(0.25, label='Octave spacing', save=True)

    include_dpoae = Bool(True)
    dpoae_window = Float(100e-3, label='DPOAE analysis window')

    frequency = Property(depends_on='start_octave, end_octave, octave_spacing, include_dpoae', save=True)

    @cached_property
    def _get_frequency(self):
        from experiment.evaluate.expr import imul
        octaves = np.arange(self.start_octave,
                            self.end_octave+self.octave_spacing,
                            self.octave_spacing, dtype=np.float)
        frequencies = (2.0**octaves)*1e3
        if self.include_dpoae:
            f2 = frequencies
            f1 = f2/1.2
            dpoae = 2*f1-f2
            frequencies = np.concatenate((f2, f1, dpoae))
            frequencies.sort()
            frequencies = imul(frequencies, 1/self.dpoae_window)
        return np.unique(frequencies)

    def _get_start_frequency(self):
        return (2**self.start_octave)*1e3

    def _get_end_frequency(self):
        return (2**self.end_octave)*1e3

    vpp = Float(1, label='Tone amplitude (peak to peak)')
    vrms = Property(depends_on='vpp', label='Tone amplitude (rms)')
    duration = Float(0.04096, label='Tone duration (Hz)')

    # Calibration results
    measured_freq = List(save=True)
    measured_spl = List(save=True)
    spl_plots = Instance(Component)
    thd_plots = Instance(Component)

    def _get_vrms(self):
        return self.vpp/np.sqrt(2)

    hardware_settings = VGroup(
        HGroup(
            Item('output'),
            Item('output_gain', label='Gain (dB)'),
        ),
        Include('mic_settings'),
        label='Hardware settings',
        show_border=True,
    )

    stimulus_settings = VGroup(
        HGroup(
            'vpp',
            Item('vrms', style='readonly', label='(rms)'),
        ),
        HGroup(
            VGroup(
                'start_octave',
                'end_octave'
            ),
            VGroup(
                'start_frequency',
                'end_frequency',
                style='readonly',
            ),
        ),
        'octave_spacing',
        'duration',
        'fft_averages',
        'waveform_averages',
        'iti',
        'trim',
        show_border=True,
        label='Tone settings',
    )

    analysis_results = VGroup(
        Tabbed(
            VGroup(
                Item('spl_plots', editor=ComponentEditor(size=(1200, 250))),
                show_labels=False,
                label='Summary',
            ),
            Item('tone_data', style='custom',
                 editor=ListEditor(use_notebook=True, deletable=False,
                                   export='DockShellWindow',
                                   page_name='.frequency'),
                 label='Tone data',
                 ),
            show_labels=False,
        ),
        style='readonly',
    )

    configuration = HSplit(
        VGroup(
            Include('hardware_settings'),
            Include('stimulus_settings'),
            enabled_when='not handler.running',
        ),
        VGroup(
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                Item('handler.current_gain', style='readonly', width=500),
                Item('handler.current_max_gain', style='readonly', width=500),
            ),
            Include('analysis_results'),
        ),
    )


class MicToneCalibration(BaseToneCalibration):

    ref_input = Enum('/Dev1/ai2', BaseToneCalibration.input_options,
                     label='Ref. mic. (channel)', save=True)

    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', save=True)
    ref_mic_sens = Float(2.685, label='Ref. mic. sens (mV/Pa)', save=True)
    ref_mic_sens_dbv = Property(depends_on='ref_mic_sens', save=True,
                                label='Ref. mic. sens. V (dB re Pa)')

    tone_data = List(Instance(MicToneCalibrationResult), ())
    measured_exp_f1 = List(save=True)
    measured_exp_f2 = List(save=True)
    measured_exp_f3 = List(save=True)
    measured_exp_thd = List(save=True)
    measured_ref_f1 = List(save=True)
    measured_ref_f2 = List(save=True)
    measured_ref_f3 = List(save=True)
    measured_ref_thd = List(save=True)
    exp_mic_sens = List(save=True)

    def _get_inputs(self):
        return ','.join([self.exp_input, self.ref_input])

    def _get_ref_mic_sens_dbv(self):
        return db(self.ref_mic_sens*1e-3)

    mic_settings = VGroup(
        HGroup(
            VGroup(
                Item('exp_input', width=10),
                Item('ref_input', width=10),
            ),
            VGroup(
                Item('exp_mic_gain', label='Gain (dB)', width=10),
                Item('ref_mic_gain', label='Gain (dB)', width=10),
            ),
        ),
        HGroup(
            'ref_mic_sens',
            Item('ref_mic_sens_dbv', style='readonly', label='V (dB re Pa)'),
        ),
        label='Microphone settings',
        show_border=True,
    )

    traits_view = View(
        Include('configuration'),
        toolbar=ToolBar(
            '-',
            Action(name='Ref. cal.', action='run_reference_calibration',
                   image=ImageResource('tool', icon_dir),
                   enabled_when='not handler.running'),
            '-',
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='not handler.running'),
            Action(name='Stop', action='stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.running'),
            '-',
            Action(name='Save', action='save',
                   image=ImageResource('document_save', icon_dir),
                   enabled_when='handler.acquired')
        ),
        resizable=True,
        height=0.95,
        width=0.5,
        id='cochlear.ToneMicCal',
    )


class MicToneCalibrationController(BaseToneCalibrationController):

    def next_gain(self, results):
        if results['exp_thd'] >= 0.01:
            self.current_max_gain = self.current_gain
            return self.current_gain - self.gain_step
        elif results['exp_mic_rms'] <= -15:
            return self.current_gain + self.gain_step
        else:
            return self.current_gain

    def analyze(self, waveforms):
        return analyze_mic_sens(
            ref_waveforms=waveforms[:, 1, :],
            exp_waveforms=waveforms[:, 0, :],
            frequency=self.current_frequency,
            vrms=self.model.vrms,
            fs=self.model.fs,
            output_gain=self.current_gain,
            ref_mic_gain=self.model.ref_mic_gain,
            exp_mic_gain=self.model.exp_mic_gain,
            ref_mic_sens=self.model.ref_mic_sens_dbv,
            trim=self.model.trim)

    def update_plots(self, waveforms, results):
        mic_psd = db(psd(waveforms, self.model.fs, 'hanning')).mean(axis=0)
        results['ref_mic_psd'] = mic_psd[1]
        results['exp_mic_psd'] = mic_psd[0]
        results['freq_psd'] = psd_freq(waveforms, self.model.fs)

        result = MicToneCalibrationResult(**results)
        frequency = results['frequency']
        ds = ArrayPlotData(freq_psd=results['freq_psd'],
                           exp_mic_psd=results['exp_mic_psd'],
                           ref_mic_psd=results['ref_mic_psd'],
                           time=results['time'],
                           exp_mic_waveform=results['exp_mic_waveform'],
                           ref_mic_waveform=results['ref_mic_waveform'])

        # Set up the waveform plot
        container = HPlotContainer(bgcolor='white', padding=10)
        plot = Plot(ds)
        plot.plot(('time', 'ref_mic_waveform'), color='black')
        plot.index_range.low_setting = self.model.trim
        plot.index_range.high_setting = 5.0/frequency+self.model.trim
        container.add(plot)
        plot = Plot(ds)
        plot.plot(('time', 'exp_mic_waveform'), color='red')
        plot.index_range.low_setting = self.model.trim
        plot.index_range.high_setting = 5.0/frequency+self.model.trim
        container.add(plot)
        result.waveform_plots = container

        # Set up the spectrum plot
        plot = Plot(ds)
        plot.plot(('freq_psd', 'ref_mic_psd'), color='black')
        plot.plot(('freq_psd', 'exp_mic_psd'), color='red')
        plot.index_scale = 'log'
        plot.title = 'Microphone response'
        plot.padding = 50
        plot.index_range.low_setting = 100
        plot.tools.append(PanTool(plot))
        zoom = ZoomTool(component=plot, tool_mode='box', always_on=False)
        plot.overlays.append(zoom)
        result.spectrum_plots = plot

        # Plot the fundamental (i.e. the tone) and first even/odd harmonics
        harmonic_container = HPlotContainer(resizable='hv', bgcolor='white',
                                            fill_padding=True, padding=10)
        for i in range(3):
            f_harmonic = results['exp_harmonics'][i]['frequency']
            plot = Plot(ds)
            plot.plot(('freq_psd', 'ref_mic_psd'), color='black')
            plot.plot(('freq_psd', 'exp_mic_psd'), color='red')
            plot.index_range.low_setting = f_harmonic-500
            plot.index_range.high_setting = f_harmonic+500
            plot.origin_axis_visible = True
            plot.padding_left = 10
            plot.padding_right = 10
            plot.border_visible = True
            plot.title = 'F{}'.format(i+1)
            harmonic_container.add(plot)
        result.harmonic_plots = harmonic_container

        self.model.tone_data.append(result)

        # Update the master overview
        self.model.measured_freq.append(results['frequency'])
        self.model.measured_spl.append(results['output_spl'])
        self.model.exp_mic_sens.append(results['exp_mic_sens'])
        for mic in ('ref', 'exp'):
            for h in range(3):
                v = results['{}_harmonics'.format(mic)][h]['mic_rms']
                name = 'measured_{}_f{}'.format(mic, h+1)
                getattr(self.model, name).append(v)
            v = results['{}_thd'.format(mic)]
            getattr(self.model, 'measured_{}_thd'.format(mic)).append(v)

        ds = ArrayPlotData(
            frequency=self.model.measured_freq,
            spl=self.model.measured_spl,
            measured_exp_thd=self.model.measured_exp_thd,
            measured_ref_thd=self.model.measured_ref_thd,
            exp_mic_sens=self.model.exp_mic_sens,
        )

        container = VPlotContainer(padding=10, bgcolor='white',
                                   fill_padding=True, resizable='hv')
        plot = Plot(ds)
        plot.plot(('frequency', 'spl'), color='black')
        plot.plot(('frequency', 'spl'), color='black', type='scatter')
        plot.index_scale = 'log'
        plot.title = 'Speaker output (dB SPL)'
        container.add(plot)

        plot = Plot(ds)
        plot.plot(('frequency', 'measured_ref_thd'), color='black')
        plot.plot(('frequency', 'measured_ref_thd'), color='black', type='scatter')
        plot.plot(('frequency', 'measured_exp_thd'), color='red')
        plot.plot(('frequency', 'measured_exp_thd'), color='red', type='scatter')
        plot.index_scale = 'log'
        plot.title = 'Total harmonic distortion (frac)'
        container.add(plot)

        plot = Plot(ds)
        plot.plot(('frequency', 'exp_mic_sens'), color='red')
        plot.plot(('frequency', 'exp_mic_sens'), color='red', type='scatter')
        plot.index_scale = 'log'
        plot.title = 'Experiment mic. sensitivity V (dB re Pa)'
        container.add(plot)

        self.model.spl_plots = container


def launch_mic_cal_gui(**kwargs):
    handler = MicToneCalibrationController()
    MicToneCalibration().edit_traits(handler=handler, **kwargs)


if __name__ == '__main__':
    #handler = MicToneCalibrationController()
    #MicToneCalibration().configure_traits(handler=handler)

    # Verify calibration
    mic_sens_dbv = db(2.685*1e-3)
    ref_cal = InterpCalibration([0, 100e3], [mic_sens_dbv, mic_sens_dbv])
    mic_file = 'c:/data/cochlear/calibration/141230 chirp calibration.mic'
    exp_cal = InterpCalibration.from_mic_file(mic_file, fixed_gain=-40)
    #for freq in (500, 1000, 2000, 4000, 8000, 16000):
    #    ref = tone_spl(freq, ref_cal, input_line='/Dev1/ai0',
    #                output_line='/Dev1/ao1', vrms=1/np.sqrt(2), gain=-20)
    #    exp = tone_spl(freq, exp_cal, input_line='/Dev1/ai1',
    #                output_line='/Dev1/ao1', vrms=1/np.sqrt(2), gain=-20)
    #    print ref, exp

    ref = tone_spl(3330, ref_cal, input_line='/Dev1/ai0',
                output_line='/Dev1/ao0', vrms=1, gain=-20)
    exp = tone_spl(3330, exp_cal, input_line='/Dev1/ai1',
                output_line='/Dev1/ao0', vrms=1, gain=-20)
    print ref+20, exp+20

    #import PyDAQmx as pyni
    #pyni.DAQmxResetDevice('Dev1')
    #ref = db(ref)
    #exp = db(exp)+18.5-20
    #print 'ref', ref
    #print 'exp', exp
    #print exp+mic_sens_dbv-ref

    ##ref = tone_power(6e3, input_line='/Dev1/ai0', output_line='/Dev1/ao1', gain=0)
    ##exp = tone_power(6e3, input_line='/Dev1/ai1', output_line='/Dev1/ao1', gain=-10)
    ##ref = db(ref)
    ##exp = db(exp)+10-20
    ##print exp+mic_sens_dbv-ref
    ###print mic_sens(4e3, '/Dev1/ai0', '/Dev1/ai1', calibration, repetitions=4,
    ###               gain=-20, vrms=1)
