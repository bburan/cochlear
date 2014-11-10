from __future__ import division

import time

import numpy as np
import matplotlib as mp
import tables

from traits.api import (HasTraits, Float, Int, Property, Enum, Bool, Instance,
                        Str, List, Array)
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
from neurogen.calibration import LinearCalibration
from neurogen.util import db, dbi
from neurogen.calibration.util import tone_power_conv, tone_power_fft, psd

import settings


def acquire_tone(frequency, vrms, waveform_averages, fft_averages, duration, ao,
                 ai, ao_fs, ai_fs, output_gain, ref_mic_gain, exp_mic_gain,
                 callback=None):

    epochs = waveform_averages*fft_averages
    calibration = LinearCalibration.as_attenuation(vrms=vrms)
    waveform = blocks.Tone(frequency=frequency, level=0)
    acq = ni.DAQmxAcquire(waveform, epochs, ao, ai, output_gain,
                          calibration=calibration, callback=callback,
                          duration=duration, adc_fs=ai_fs, dac_fs=ao_fs)
    acq.start()
    acq.join()


def analyze_tone(waveforms, frequency, vrms, waveform_averages, fft_averages,
                 ai_fs, output_gain, ref_mic_gain, exp_mic_gain, ref_mic_sens,
                 exp_mic_sens=None, trim=0, thd_harmonics=2):

    trim_n = int(trim*ai_fs)
    waveforms = waveforms[:, :, trim_n:-trim_n]
    c, t = waveforms.shape[-2:]

    # Average time samples
    waveforms = waveforms.reshape((waveform_averages, fft_averages, c, t))
    waveforms = waveforms.mean(axis=0)

    # Get average PSD
    s_freq, s_psd = psd(waveforms, ai_fs, window='flattop')
    exp_psd, ref_psd = db(s_psd).mean(axis=0)

    # Get average tone power across channels
    power = tone_power_conv(waveforms, ai_fs, frequency, window='flattop')
    exp_power, ref_power = db(power).mean(axis=0)

    exp_waveform, ref_waveform = waveforms.mean(axis=0)
    time = np.arange(len(exp_waveform))/ai_fs

    # Correct for gains (i.e. we want to know the *actual* Vrms at 0 dB input
    # and 0 dB output gain).
    exp_psd -= exp_mic_gain
    exp_power -= exp_mic_gain
    ref_psd -= ref_mic_gain
    ref_power -= ref_mic_gain

    max_harmonic = int(np.floor((ai_fs/2.0)/frequency))
    harmonics = []
    for i in range(1, max_harmonic+1):
        f_harmonic = frequency*i
        p = tone_power_conv(waveforms, ai_fs, f_harmonic, window='flattop')
        exp_harmonic, ref_harmonic = db(p).mean(axis=0)
        harmonics.append({
            'harmonic': i,
            'frequency': f_harmonic,
            'exp_mic_rms': exp_harmonic,
            'ref_mic_rms': ref_harmonic
        })

    ref_harmonic_v = []
    exp_harmonic_v = []
    for h_info in harmonics:
        ref_harmonic_v.append(dbi(h_info['ref_mic_rms']))
        exp_harmonic_v.append(dbi(h_info['exp_mic_rms']))
    ref_harmonic_v = np.asarray(ref_harmonic_v)[:(thd_harmonics+1)]
    exp_harmonic_v = np.asarray(exp_harmonic_v)[:(thd_harmonics+1)]
    ref_thd = (np.sum(ref_harmonic_v[1:]**2)**0.5)/ref_harmonic_v[0]
    exp_thd = (np.sum(exp_harmonic_v[1:]**2)**0.5)/exp_harmonic_v[0]

    # Actual output SPL
    output_spl = ref_power-ref_mic_sens-db(20e-6)

    # Output SPL assuming 0 dB gain and 1 VRMS
    norm_output_spl = output_spl-output_gain-db(vrms)

    exp_mic_sens = exp_power+ref_mic_sens-ref_power

    return {
        'frequency': frequency,
        'time': time,
        'freq_psd': s_freq,
        'ref_mic_rms': ref_power,
        'ref_mic_psd': ref_psd,
        'ref_thd': ref_thd,
        'ref_mic_waveform': ref_waveform,
        'exp_mic_rms': exp_power,
        'exp_mic_psd': exp_psd,
        'exp_thd': exp_thd,
        'exp_mic_waveform': exp_waveform,
        'output_spl': output_spl,
        'norm_output_spl': norm_output_spl,
        'harmonics': harmonics,
        'exp_mic_sens': exp_mic_sens,
        'waveforms': waveforms,
    }


class ToneCalibrationResult(HasTraits):

    frequency = Float(save=True)
    time = Array(save=True)
    freq_psd = Array(save=True)
    ref_mic_rms = Float(save=True)
    ref_mic_psd = Array(save=True)
    ref_thd = Float(save=True)
    ref_mic_waveform = Array(save=True)
    exp_mic_rms = Float(save=True)
    exp_mic_psd = Array(save=True)
    exp_thd = Float(save=True)
    exp_mic_waveform = Array(save=True)
    output_spl = Float(save=True)
    norm_output_spl = Float(save=True)
    exp_mic_sens = Float(save=True)
    waveforms = Array(save=True)

    harmonics = List()

    waveform_plots = Instance(Component)
    spectrum_plots = Instance(Component)
    harmonic_plots = Instance(Component)

    traits_view = View(
        VGroup(
            HGroup(
                'exp_mic_rms',
                Item('exp_thd', label='THD (frac)'),
            ),
            HGroup(
                'ref_mic_rms',
                Item('ref_thd', label='THD (frac)'),
            ),
            'output_spl',
            VGroup(
                Item('waveform_plots', editor=ComponentEditor(size=(600, 250))),
                Item('spectrum_plots', editor=ComponentEditor(size=(600, 250))),
                Item('harmonic_plots', editor=ComponentEditor(size=(600, 250))),
                show_labels=False,
            ),
            style='readonly',
        )
    )


class ToneCalibrationController(Controller):

    # Is handler currently acquiring data?
    running = Bool(False)

    # Has data been successfully acquired (and ready for save)?
    acquired = Bool(False)

    epochs_acquired = Int(0)
    iface_daq = Instance('cochlear.nidaqmx.DAQmxAcquire')
    model = Instance('ToneCalibration')
    frequencies = List
    current_frequency = Float(label='Current frequency (Hz)')

    def run_reference_calibration(self, info=None):
        pass

    def setup(self):
        self.running = True
        self.acquired = False
        self.epochs_acquired = 0
        calibration = LinearCalibration.as_attenuation(vrms=self.model.vrms)
        waveform = blocks.Tone(frequency=self.current_frequency, level=0)
        epochs = self.model.waveform_averages*self.model.fft_averages
        self.iface_daq = ni.DAQmxAcquire(waveform,
                                         epochs,
                                         self.model.output,
                                         self.model.inputs,
                                         self.model.output_gain,
                                         calibration=calibration,
                                         callback=self.update_status,
                                         duration=self.model.duration,
                                         adc_fs=self.model.adc_fs,
                                         dac_fs=self.model.dac_fs)

    def next_frequency(self):
        if self.frequencies:
            self.current_frequency = self.frequencies.pop(0)
            self.setup()
            self.iface_daq.start()
        else:
            self.acquired = True
            self.running = False

    def update_status(self, acquired, done):
        self.epochs_acquired = acquired
        if done:
            results = analyze_tone(
                waveforms=self.iface_daq.waveforms,
                frequency=self.current_frequency,
                vrms=self.model.vrms,
                waveform_averages=self.model.waveform_averages,
                fft_averages=self.model.fft_averages,
                ai_fs=self.model.adc_fs,
                output_gain=self.model.output_gain,
                ref_mic_gain=self.model.ref_mic_gain,
                exp_mic_gain=self.model.exp_mic_gain,
                ref_mic_sens=self.model.ref_mic_sens_dbv,
                trim=self.model.trim)
            self.update_plots(results)
            self.next_frequency()

    def update_plots(self, results):
        result = ToneCalibrationResult(**results)

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
        plot.index_range.high_setting = 5.0/self.current_frequency
        container.add(plot)
        plot = Plot(ds)
        plot.plot(('time', 'exp_mic_waveform'), color='red')
        plot.index_range.high_setting = 5.0/self.current_frequency
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
            f_harmonic = results['harmonics'][i]['frequency']
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
                v = results['harmonics'][h]['{}_mic_rms'.format(mic)]
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
                print trait
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


class ToneCalibration(HasTraits):

    # Calibration settings
    adc_fs = Float(200e3, label='Analog input sampling rate (Hz)', save=True)
    dac_fs = Float(200e3, label='Analog output sampling rate (Hz)', save=True)

    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', save=True)
    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', save=True)
    ref_mic_sens = Float(2.66, label='Ref. mic. sens (mV/Pa)', save=True)
    ref_mic_sens_dbv = Property(depends_on='ref_mic_sens', save=True,
                                label='Ref. mic. sens. V (dB re Pa)')

    input_options = ['/Dev1/ai{}'.format(i) for i in range(4)]
    output = Enum(('/Dev1/ao0', '/Dev1/ao1'), label='Output (channel)', save=True)
    exp_input = Enum('/Dev1/ai1', input_options, label='Exp. mic. (channel)', save=True)
    ref_input = Enum('/Dev1/ai2', input_options, label='Ref. mic. (channel)', save=True)
    inputs = Property(depends_on='exp_input, ref_input', save=True)
    output_gain = Float(31.5, label='Output gain (dB)', save=True)

    waveform_averages = Int(2, label='Number of tones per FFT', save=True)
    fft_averages = Int(2, label='Number of FFTs', save=True)
    iti = Float(0.001, label='Inter-tone interval', save=True)
    trim = Float(0.01, label='Trim (sec)', save=True)
    trim_n = Property(depends_on='trim, adc_fs', save=True)

    start_octave = Float(-2, label='Start octave', save=True)
    start_frequency = Property(depends_on='start_octave', label='End octave', save=True)
    end_octave = Float(5, save=True)
    end_frequency = Property(depends_on='end_octave', save=True)
    octave_spacing = Float(1, label='Octave spacing', save=True)

    frequency = Property(depends_on='start_octave, end_octave, octave_spacing', save=True)

    def _get_frequency(self):
        octaves = np.arange(self.start_octave,
                            self.end_octave+self.octave_spacing,
                            self.octave_spacing, dtype=np.float)
        return (2.0**octaves)*1e3

    def _get_start_frequency(self):
        return (2**self.start_octave)*1e3

    def _get_end_frequency(self):
        return (2**self.end_octave)*1e3

    vpp = Float(10, label='Tone amplitude (peak to peak)')
    vrms = Property(depends_on='vpp', label='Tone amplitude (rms)')
    duration = Float(0.1, label='Tone duration (Hz)')

    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Number of repeats')

    # Calibration results
    ref_mic_rms = Float(label='Ref. mic. power (dB re V)')
    exp_mic_rms = Float(label='Exp. mic. power (dB re V)')
    ref_thd = Float(label='THD in ref. signal (frac)')
    exp_thd = Float(label='THD in exp. signal (frac)')
    output_spl = Float(label='Speaker output (dB SPL)')

    #waveform_plots = Instance(Component)
    #spectrum_plots = Instance(Component)
    #harmonic_plots = Instance(Component)

    tone_data = List(Instance(ToneCalibrationResult), ())

    measured_freq = List(save=True)
    measured_spl = List(save=True)
    measured_exp_f1 = List(save=True)
    measured_exp_f2 = List(save=True)
    measured_exp_f3 = List(save=True)
    measured_exp_thd = List(save=True)
    measured_ref_f1 = List(save=True)
    measured_ref_f2 = List(save=True)
    measured_ref_f3 = List(save=True)
    measured_ref_thd = List(save=True)
    exp_mic_sens = List(save=True)

    spl_plots = Instance(Component)
    thd_plots = Instance(Component)

    def _get_ref_mic_sens_dbv(self):
        return db(self.ref_mic_sens*1e-3)

    def _get_inputs(self):
        return ','.join([self.exp_input, self.ref_input])

    def _get_vrms(self):
        return self.vpp/np.sqrt(2)

    def _get_averages(self):
        return self.fft_averages*self.waveform_averages

    def _get_trim_n(self):
        return int(self.trim*self.adc_fs)

    hardware_settings = VGroup(
        HGroup(
            Item('output'),
            Item('output_gain', label='Gain (dB)'),
        ),
        HGroup(
            Item('exp_input'),
            Item('exp_mic_gain', label='Gain (dB)'),
        ),
        HGroup(
            Item('ref_input'),
            Item('ref_mic_gain', label='Gain (dB)'),
        ),
        label='Hardware settings',
        show_border=True,
    )

    stimulus_settings = VGroup(
        Item('vrms', style='readonly'),
        'vpp',
        HGroup('start_octave', 'start_frequency'),
        HGroup('end_octave', 'end_frequency'),
        'octave_spacing',
        'duration',
        'fft_averages',
        'waveform_averages',
        'iti',
        'trim',
        Item('averages', style='readonly'),
        show_border=True,
        label='Tone settings',
    )

    mic_settings = VGroup(
        'ref_mic_sens',
        Item('ref_mic_sens_dbv', style='readonly'),
        label='Microphone settings',
        show_border=True,
    )

    analysis_results = VGroup(
        Tabbed(
            Item('tone_data', style='custom',
                 editor=ListEditor(use_notebook=True, deletable=False,
                                   export='DockShellWindow',
                                   page_name='.frequency'),
                 ),
            VGroup(
                Item('spl_plots', editor=ComponentEditor(size=(1200, 250))),
                show_labels=False,
            ),
            show_labels=False,
        ),
        style='readonly',
    )

    traits_view = View(
        HSplit(
            VGroup(
                Include('hardware_settings'),
                Include('mic_settings'),
                Include('stimulus_settings'),
                enabled_when='not handler.running',
            ),
            VGroup(
                HGroup(
                    Item('handler.epochs_acquired', style='readonly'),
                ),
                Include('analysis_results'),
            ),
        ),
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
        width=0.95,
        id='cochlear.ToneCal',
    )


if __name__ == '__main__':
    ToneCalibration().configure_traits(handler=ToneCalibrationController())
