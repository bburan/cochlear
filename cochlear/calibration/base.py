from __future__ import division

import logging
log = logging.getLogger(__name__)

import tables
import numpy as np

from traits.api import (Enum, Float, Int, Property, HasTraits, Instance, Bool,
                        Str)
from traitsui.api import (VGroup, View, Include, Item, HSplit, Tabbed, ToolBar,
                          Action, ActionGroup, MenuBar, Menu)
from pyface.api import ImageResource, error
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot

from experiment import AbstractParadigm, AbstractController, icon_dir
from experiment.util import get_save_file

from cochlear import nidaqmx as ni
from cochlear.calibration import standard
from cochlear import settings


class BaseSignalSettings(AbstractParadigm):

    kw = dict(context=True)

    output = Enum(('ao0', 'ao1'), label='Analog output (channel)', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vpp)', **kw)
    output_gain = Float(0, label='Output gain (dB)', **kw)
    fft_averages = Int(4, label='Number of FFTs', **kw)
    discard = Int(0, label='Discard repetitions', **kw)
    waveform_averages = Int(4, label='Number of averages per FFT', **kw)
    iti = Float(0.001, label='Inter-chirp interval', **kw)
    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Total repetitions', **kw)

    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', **kw)
    exp_range = Float(1, label='Expected input range (Vpp)', **kw)

    def _get_averages(self):
        return self.fft_averages*self.waveform_averages + self.discard

    presentation_settings = VGroup(
        'fft_averages',
        'waveform_averages',
        'iti',
        label='Presentation settings',
        show_border=True,
    )

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

    traits_view = View(
        VGroup(
            Include('output_settings'),
            Include('mic_settings'),
            Include('stimulus_settings'),
            Include('presentation_settings'),
        )
    )


class ReferenceSettingsMixin(HasTraits):

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


class HRTFSettingsMixin(HasTraits):

    mic_settings = VGroup(
        'exp_mic_gain',
        'exp_range',
        label='Microphone settings',
        show_border=True,
    )


class BaseSignalExperiment(HasTraits):

    paradigm = Instance(BaseSignalSettings, ())

    plot_data = Instance(ArrayPlotData)
    sig_waveform_plot = Instance(Component)
    mic_waveform_plot = Instance(Component)
    mic_psd_plot = Instance(Component)
    mic_phase_plot = Instance(Component)
    sig_psd_plot = Instance(Component)
    sig_phase_plot = Instance(Component)
    speaker_spl_plot = Instance(Component)
    exp_mic_sens_plot = Instance(Component)

    KEYS = ('time', 'sig_waveform', 'exp_mic_waveform', 'ref_mic_waveform',
            'frequency', 'ref_mic_psd', 'exp_mic_psd', 'sig_psd', 'speaker_spl',
            'exp_mic_sens', 'exp_mic_phase', 'ref_mic_phase', 'sig_phase')

    def update_plots(self, results, freq_lb=500, freq_ub=50e3):
        for k in self.KEYS:
            if k in results:
                self.plot_data.set_data(k, results[k][1:])
        for plot in (self.mic_psd_plot, self.mic_phase_plot, self.sig_psd_plot,
                     self.sig_phase_plot, self.exp_mic_sens_plot,
                     self.speaker_spl_plot):
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
        return plot

    def _mic_phase_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Microphone Phase')
        plot.plot(('frequency', 'ref_mic_phase'), index_scale='log',
                  color='black')
        plot.plot(('frequency', 'exp_mic_phase'), index_scale='log', color='red')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Phase (radians)'
        return plot

    def _speaker_spl_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Speaker Output')
        plot.plot(('frequency', 'speaker_spl'), index_scale='log',
                  color='black')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Power (dB SPL)'
        return plot

    def _sig_psd_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Signal Spectrum')
        plot.plot(('frequency', 'sig_psd'), index_scale='log', color='black')
        plot.index_axis.title = 'Frequency (Hz)'
        plot.value_axis.title = 'Power (dB)'
        return plot

    def _sig_phase_plot_default(self):
        plot = Plot(self.plot_data, padding=[75, 25, 25, 50],
                    title='Signal Spectrum')
        plot.plot(('frequency', 'sig_phase'), index_scale='log', color='black')
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

    response_plots = VGroup(
        Item('mic_waveform_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        Item('mic_psd_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        Item('mic_phase_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        Item('exp_mic_sens_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        label='Mic PSD',
    )

    signal_plots = VGroup(
        Item('sig_waveform_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        Item('sig_psd_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        Item('sig_phase_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        Item('speaker_spl_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        label='Signal',
    )

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                Tabbed(
                    Include('response_plots'),
                    Include('signal_plots'),
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


class BaseSignalController(AbstractController):

    adc_fs = 200e3
    dac_fs = 200e3
    epochs_acquired = Int(0)
    complete = Bool(False)
    SAVE_PATTERN = 'Microphone calibration (*.mic)|*.mic'
    SAVE_DIRECTORY = settings.CALIBRATION_DIR

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

    def save(self, info=None):
        filename = get_save_file(self.SAVE_DIRECTORY, self.SAVE_PATTERN)
        if filename is not None:
            save_results(filename, self.results, self.result_settings)
        self.complete = False


class ReferenceControllerMixin(HasTraits):

    MIC_INPUT = '{}, {}'.format(ni.DAQmxDefaults.MIC_INPUT,
                                ni.DAQmxDefaults.REF_MIC_INPUT)


class HRTFControllerMixin(HasTraits):

    MIC_INPUT = ni.DAQmxDefaults.MIC_INPUT
    calibration = Instance('neurogen.calibration.Calibration')
    filename = Str()

    def save(self, info=None):
        save_results(self.filename, self.results, self.result_settings)
        self.complete = False


class HRTFExperimentMixin(HasTraits):

    response_plots = VGroup(
        Item('mic_waveform_plot', editor=ComponentEditor(), width=500,
             height=200, show_label=False),
        Item('mic_psd_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        Item('mic_phase_plot', editor=ComponentEditor(), width=500, height=200,
             show_label=False),
        label='Mic response',
    )


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
