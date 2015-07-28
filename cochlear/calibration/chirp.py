from __future__ import division

import logging
log = logging.getLogger(__name__)

import shutil

from pyface.api import ImageResource, error
from traits.api import (Str, Float, Int, Property, Bool, Enum, HasTraits,
                        Instance)
from traitsui.api import (View, Item, VGroup, Include, ToolBar, HSplit, Tabbed)
from traitsui.menu import MenuBar, Menu, ActionGroup, Action
from enable.api import Component, ComponentEditor
from chaco.api import (ArrayPlotData, Plot)

import tables
import numpy as np

from experiment import (icon_dir, AbstractController, AbstractParadigm)
from experiment.util import get_save_file

from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear.calibration import ChirpCalibration, standard
from cochlear import settings


################################################################################
# Base classes supporting both chirp cal tests
################################################################################
def save_results(filename, results, settings):
    with tables.open_file(filename, 'w') as fh:
        fs = results['fs']
        for k, v in results.items():
            if np.iterable(v):
                node = fh.create_array(fh.root, k, v)
                if k.endswith('waveform'):
                    node._v_attrs['fs'] = fs
            else:
                fh.root._v_attrs[k] = v
        for k, v in settings.items():
            fh.root._v_attrs[k] = v


class BaseChirpSettings(AbstractParadigm):

    kw = dict(context=True)

    output = Enum(('ao1', 'ao0'), label='Analog output (channel)', **kw)
    rise_time = Float(5e-3, label='Envelope rise time', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    output_gain = Float(0, label='Output gain (dB)', **kw)
    freq_lb = Float(0.5e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(50e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    fft_averages = Int(4, label='Number of FFTs', **kw)
    waveform_averages = Int(4, label='Number of chirps per FFT', **kw)
    iti = Float(0.001, label='Inter-chirp interval', **kw)
    start_attenuation = Float(0, label='Start attenuation (dB)', **kw)
    end_attenuation = Float(0, label='End attenuation (dB)', **kw)

    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', **kw)
    exp_range = Float(1, label='Expected input range (Vpp)', **kw)

    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Number of chirps', **kw)
    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, iti',
                              label='Total duration (sec)')

    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.duration+self.iti)*self.averages

    def _get_averages(self):
        return self.fft_averages*self.waveform_averages

    output_settings = VGroup(
        'output',
        'output_gain',
        'amplitude',
        label='Output settings',
        show_border=True,
    )

    mic_settings = VGroup(
        'exp_mic_gain',
        'exp_range',
        label='Microphone settings',
        show_border=True,
    )

    stimulus_settings = VGroup(
        'rise_time',
        'freq_lb',
        'freq_ub',
        'freq_resolution',
        'fft_averages',
        'waveform_averages',
        'iti',
        Item('averages', style='readonly'),
        Item('duration', style='readonly'),
        Item('total_duration', style='readonly'),
        'start_attenuation',
        'end_attenuation',
        show_border=True,
        label='Chirp settings',
    )

    traits_view = View(
        VGroup(
            Include('output_settings'),
            Include('mic_settings'),
            Include('stimulus_settings'),
        )
    )


class BaseChirpExperiment(HasTraits):

    paradigm = Instance(BaseChirpSettings, ())

    plot_data = Instance(ArrayPlotData)
    sig_waveform_plot = Instance(Component)
    mic_waveform_plot = Instance(Component)
    mic_psd_plot = Instance(Component)
    sig_psd_plot = Instance(Component)
    speaker_spl_plot = Instance(Component)
    exp_mic_sens_plot = Instance(Component)

    KEYS = ('time', 'sig_waveform', 'exp_mic_waveform', 'ref_mic_waveform',
            'frequency', 'ref_mic_psd', 'exp_mic_psd', 'sig_psd', 'speaker_spl',
            'exp_mic_sens')

    def update_plots(self, results, freq_lb=500, freq_ub=50e3):
        for k in self.KEYS:
            if k in results:
                self.plot_data.set_data(k, results[k])
        for plot in (self.mic_psd_plot, self.sig_psd_plot,
                     self.exp_mic_sens_plot, self.speaker_spl_plot):
            plot.index_mapper.range.low_setting = freq_lb*0.9
            plot.index_mapper.range.high_setting = freq_ub*1.1

    def _plot_data_default(self):
        pd = ArrayPlotData()
        for k in self.KEYS:
            pd.set_data(k, [])
        return pd

    def _sig_waveform_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Signal Waveform')
        plot.plot(('time', 'sig_waveform'), color='black')
        plot.index_axis.title = 'Time (sec)'
        plot.value_axis.title = 'Signal (V)'
        return plot

    def _mic_waveform_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Microphone Response')
        plot.plot(('time', 'ref_mic_waveform'), color='black')
        plot.plot(('time', 'exp_mic_waveform'), color='red', alpha=0.5)
        plot.index_axis.title = 'Time (sec)'
        plot.value_axis.title = 'Signal (V)'
        return plot

    def _mic_psd_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Microphone Spectrum')
        plot.plot(('frequency', 'ref_mic_psd'), index_scale='log',
                  color='black')
        plot.plot(('frequency', 'exp_mic_psd'), index_scale='log', color='red')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Power (dB)'
        plot.value_mapper.range.low_setting = -100
        return plot

    def _speaker_spl_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Speaker Output')
        plot.plot(('frequency', 'speaker_spl'), index_scale='log',
                  color='black')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Power (dB SPL)'
        plot.value_mapper.range.low_setting = 40
        return plot

    def _sig_psd_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Signal Spectrum')
        plot.plot(('frequency', 'sig_psd'), index_scale='log', color='black')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Power (dB)'
        return plot

    def _exp_mic_sens_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Experiment Microphone Sensitivity')
        plot.plot(('frequency', 'exp_mic_sens'), index_scale='log',
                  color='black')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Sens.'
        return plot

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                Tabbed(
                    Include('response_plots'),
                    VGroup(
                        Item('sig_waveform_plot', editor=ComponentEditor(),
                             width=500, height=200, show_label=False),
                        Item('sig_psd_plot', editor=ComponentEditor(),
                             width=500, height=200, show_label=False),
                        Item('speaker_spl_plot', editor=ComponentEditor(),
                             width=500, height=200, show_label=False),
                        label='Signal',
                    )
                ),
            ),
            show_labels=False,
        ),
        toolbar=ToolBar(
            '-',
            Action(name='Ref. cal.', action='run_reference_calibration',
                   image=ImageResource('tool', icon_dir),
                   enabled_when='not handler.state=="running"'),
            '-',
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='not handler.state=="running"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            '-',
            Action(name='Save', action='save',
                   image=ImageResource('document_save', icon_dir),
                   enabled_when='handler.complete')
        ),
        resizable=True,
        menubar=MenuBar(
            Menu(
                ActionGroup(
                    Action(name='Load settings', action='load_settings'),
                    Action(name='Save settings', action='save_settings'),
                ),
                name='&Settings',
            ),
        ),
    )


class BaseChirpController(AbstractController):

    adc_fs = 200e3
    dac_fs = 200e3
    epochs_acquired = Int(0)
    complete = Bool(False)

    def setup_acquire(self):
        # Load the variables
        output = self.get_current_value('output')
        analog_output = '/{}/{}'.format(ni.DAQmxDefaults.DEV, output)
        self.iface = ChirpCalibration(
            freq_lb=self.get_current_value('freq_lb'),
            freq_ub=self.get_current_value('freq_ub'),
            start_atten=self.get_current_value('start_attenuation'),
            end_atten=self.get_current_value('end_attenuation'),
            vrms=self.get_current_value('amplitude'),
            gain=self.get_current_value('output_gain'),
            repetitions=self.get_current_value('averages'),
            duration=self.get_current_value('duration'),
            rise_time=self.get_current_value('rise_time'),
            iti=self.get_current_value('iti'),
            fs=self.adc_fs,
            output_line=analog_output,
            input_line=self.MIC_INPUT,
            callback=self.poll,
        )

    def start(self, info=None):
        log.debug('Starting calibration')
        try:
            self.complete = False
            self.epochs_acquired = 0
            self.complete = False
            self.calibration_accepted = False

            self.initialize_context()
            self.refresh_context()
            self.setup_acquire()
            self.iface.acquire(join=False)
            self.state = 'running'
        except Exception as e:
            raise
            if info is not None:
                error(info.ui.control, str(e))
            else:
                error(None, str(e))
            self.stop(info)

    def stop(self, info=None):
        self.state = 'halted'

    def run_reference_calibration(self, info=None):
        standard.launch_gui(parent=info.ui.control, kind='livemodal')

    def save_settings(self, info=None):
        self.save_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')

    def load_settings(self, info=None):
        self.load_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')

    def poll(self, epochs_acquired, complete):
        self.epochs_acquired = epochs_acquired
        if complete:
            self.finalize()
            self.stop()


################################################################################
# Reference microphone calibration
################################################################################
class ReferenceCalibrationSettings(BaseChirpSettings):

    kw = dict(context=True)
    ref_mic_sens_mv = Float(2.703, label='Ref. mic. sens. (mV/Pa)', **kw)
    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', **kw)
    ref_mic_sens = Property(depends_on='ref_mic_sens_mv',
                            label='Ref. mic. sens. dB(mV/Pa)', **kw)

    def _get_ref_mic_sens(self):
        return self.ref_mic_sens_mv*1e-3

    mic_settings = VGroup(
        'ref_mic_sens_mv',
        'ref_mic_gain',
        'exp_mic_gain',
        'exp_range',
        label='Microphone settings',
        show_border=True,
    )


class ReferenceCalibrationController(BaseChirpController):

    MIC_INPUT = '{}, {}'.format(ni.DAQmxDefaults.MIC_INPUT,
                                ni.DAQmxDefaults.REF_MIC_INPUT)
    SAVE_PATTERN = 'Microphone calibration (*.mic)|*.mic'
    SAVE_DIRECTORY = settings.CALIBRATION_DIR

    def save(self, info=None):
        filename = get_save_file(self.SAVE_DIRECTORY, self.SAVE_PATTERN)
        if filename is not None:
            save_results(filename, self.results, self.result_settings)
        self.complete = False

    def finalize(self):
        ref_mic_sens = self.get_current_value('ref_mic_sens')
        ref_mic_gain = dbi(self.get_current_value('ref_mic_gain'))
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=[exp_mic_gain, ref_mic_gain])

        exp_mic_psd, ref_mic_psd = db(results['mic_psd'])
        exp_mic_waveform, ref_mic_waveform = \
            results['mic_waveforms'].mean(axis=0)
        results['exp_mic_waveform'] = exp_mic_waveform
        results['ref_mic_waveform'] = ref_mic_waveform
        results['ref_mic_psd'] = ref_mic_psd
        results['exp_mic_psd'] = exp_mic_psd
        results['exp_mic_sens'] = exp_mic_psd+db(ref_mic_sens)-ref_mic_psd
        results['speaker_spl'] = ref_mic_psd-db(ref_mic_sens)-db(20e-6)
        results['frequency'] = results['mic_frequency']

        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        self.model.update_plots(results, freq_lb, freq_ub)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class ReferenceCalibration(BaseChirpExperiment):

    response_plots = VGroup(
        Item('mic_waveform_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        Item('mic_psd_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        Item('exp_mic_sens_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        label='Mic response',
    )


def reference_calibration(**kwargs):
    controller = ReferenceCalibrationController()
    paradigm = ReferenceCalibrationSettings(output='ao0')
    ReferenceCalibration(paradigm=paradigm) \
        .configure_traits(handler=controller, **kwargs)


################################################################################
# Ear transfer function
################################################################################
class HRTFSettings(BaseChirpSettings):

    mic_settings = VGroup(
        'exp_mic_gain',
        'exp_range',
        label='Microphone settings',
        show_border=True,
    )


class HRTFController(BaseChirpController):

    MIC_INPUT = ni.DAQmxDefaults.MIC_INPUT
    calibration = Instance('neurogen.calibration.Calibration')
    filename = Str()

    def save(self, info=None):
        save_results(self.filename, self.results, self.result_settings)
        self.complete = False

    def finalize(self):
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=exp_mic_gain)
        exp_mic_waveform = results['mic_waveforms'].mean(axis=0)[0]
        exp_mic_psd = db(results['mic_psd'][0])

        speaker_spl = self.calibration.get_spl(results['mic_frequency'],
                                               results['mic_psd'][0])

        results['exp_mic_waveform'] = exp_mic_waveform
        results['exp_mic_psd'] = exp_mic_psd
        results['frequency'] = results['mic_frequency']
        results['speaker_spl'] = speaker_spl

        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        self.model.update_plots(results, freq_lb, freq_ub)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class HRTF(BaseChirpExperiment):

    response_plots = VGroup(
        Item('mic_waveform_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        Item('mic_psd_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        label='Mic response',
    )


def hrtf_calibration(calibration, filename, **kwargs):
    controller = HRTFController(calibration=calibration, filename=filename)
    paradigm = HRTFSettings()
    HRTF(paradigm=paradigm).configure_traits(handler=controller, **kwargs)


if __name__ == '__main__':
    reference_calibration()
    #import os.path
    #from neurogen.calibration import InterpCalibration
    #mic_file = os.path.join('c:/data/cochlear/calibration',
    #                        '150407 - calibration with 377C10.mic')
    #c = InterpCalibration.from_mic_file(mic_file)
    #hrtf_calibration(c, 'temp.hdf5')
