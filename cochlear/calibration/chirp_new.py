from __future__ import division

import logging
log = logging.getLogger(__name__)

import shutil

import os.path
import numpy as np
import tables

from pyface.api import ImageResource, error
from traits.api import (Float, Int, Property, Any, Bool, Enum, HasTraits,
                        Instance)
from traitsui.api import (View, Item, VGroup, Include, ToolBar, HSplit)
from traitsui.menu import MenuBar, Menu, ActionGroup, Action
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer, ArrayPlotData, Plot)
from chaco.tools.api import PanTool, BetterSelectingZoom

from experiment import (icon_dir, AbstractController, AbstractParadigm,
                        AbstractData)
from experiment.util import get_save_file
from experiment.channel import FileFilteredEpochChannel

from neurogen import generate_waveform, prepare_for_write
from neurogen import block_definitions as blocks
from neurogen.calibration import InterpCalibration, FlatCalibration
from neurogen.calibration.util import psd, psd_freq
from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear.calibration import get_chirp_transform
from cochlear.calibration import standard
from cochlear import settings

DAC_FS = 200e3
ADC_FS = 200e3


class BaseChirpCalSettings(AbstractParadigm):

    kw = dict(context=True)

    output = Enum(('ao1', 'ao0'), label='Analog output (channel)', **kw)
    rise_time = Float(5e-3, label='Envelope rise time', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    output_gain = Float(6, label='Output gain (dB)', **kw)
    freq_lb = Float(0.5e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(50e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    fft_averages = Int(4, label='Number of FFTs', **kw)
    waveform_averages = Int(4, label='Number of chirps per FFT', **kw)
    iti = Float(0.01, label='Inter-chirp interval', **kw)
    start_attenuation = Float(30, label='Start attenuation (dB)', **kw)
    end_attenuation = Float(-30, label='End attenuation (dB)', **kw)

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


class EarChirpCalSettings(BaseChirpCalSettings):

    mic_settings = VGroup(
        'exp_mic_gain',
        'exp_range',
        label='Microphone settings',
        show_border=True,
    )


class ChirpCalSettings(BaseChirpCalSettings):

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


class BaseChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    exp_mic_sens = Any()

    def _create_microphone_node(self, fs, epoch_duration, which='exp'):
        node_name = '{}_microphone'.format(which)
        if node_name in self.fh.root:
            self.fh.root.get_child(node_name).remove()
        filter_kw = dict(filter_freq_hp=5, filter_freq_lp=80e3,
                         filter_btype='bandpass', filter_order=1)
        node = FileFilteredEpochChannel(node=self.fh.root, name=node_name,
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        setattr(self, node_name, node)

    def _create_array(self, name, array, store_node=None):
        if store_node is None:
            store_node = self.store_node
        if name in store_node:
            store_node._f_get_child(name).remove()
        return self.fh.create_array(store_node, name, array)


class EarChirpCalData(BaseChirpCalData):

    def compute_transfer_functions(self, waveform_fs, waveform, exp_mic_gain,
                                   waveform_averages, exp_mic_cal):

        fft_window = 'boxcar'
        self.time = np.arange(len(waveform))/waveform_fs
        self.waveform = waveform

        # Get frequencies
        frequency = psd_freq(self.exp_microphone[:], self.exp_microphone.fs)
        exp_psd = psd(self.exp_microphone[:], self.exp_microphone.fs,
                      fft_window, waveform_averages)
        exp_psd = db(exp_psd.mean(axis=0))-exp_mic_gain
        exp_psd_vrms = dbi(exp_psd)

        self.frequency = frequency
        self.exp_mic_psd = exp_psd

        # Actual power in signal
        self.sig_frequency = psd_freq(waveform, waveform_fs)
        self.sig_psd = db(psd(waveform, waveform_fs, window=fft_window))

        self.speaker_spl = exp_mic_cal.get_spl(frequency, exp_psd_vrms)
        self.speaker_spl_norm = self.speaker_spl-self.sig_psd

        # Save the data in the HDF5 file
        self._create_array('frequency', self.frequency)
        self._create_array('exp_psd_rms', self.exp_mic_psd)
        self._create_array('time', self.time)
        self._create_array('waveform', self.waveform)


class ChirpCalData(BaseChirpCalData):

    def compute_transfer_functions(self, waveform_fs, waveform, ref_mic_sens,
                                   ref_mic_gain, exp_mic_gain,
                                   waveform_averages):

        fft_window = 'boxcar'
        self.time = np.arange(len(waveform))/waveform_fs
        self.waveform = waveform

        # All functions are computed using these frequencies
        self.frequency = self.ref_microphone.get_fftfreq()

        # Get frequencies
        self.frequency = psd_freq(self.ref_microphone[:],
                                  self.ref_microphone.fs)

        # Compute PSD of microphone signals, convert to dB re 1V and compensate
        # for measurement gain settings
        args = fft_window, waveform_averages
        ref_psd = psd(self.ref_microphone[:], self.ref_microphone.fs, *args)
        exp_psd = psd(self.exp_microphone[:], self.exp_microphone.fs, *args)
        self.ref_mic_psd = db(ref_psd.mean(axis=0))-ref_mic_gain
        self.exp_mic_psd = db(exp_psd.mean(axis=0))-exp_mic_gain

        # Sensitivity of experiment microphone as function of frequency
        # expressed as Vrms (dB re Pa).  This is equivalent to
        # (Vprobe/Vcal)/Ccal
        self.exp_mic_sens = self.exp_mic_psd+db(ref_mic_sens)-self.ref_mic_psd

        # Actual output of speaker
        self.speaker_spl = self.ref_mic_psd-db(ref_mic_sens)-db(20e-6)

        # Actual power in signal
        self.sig_frequency = psd_freq(waveform, waveform_fs)
        self.sig_psd = psd(waveform, waveform_fs, window=fft_window)

        # Save the data in the HDF5 file
        self._create_array('frequency', self.frequency)
        self._create_array('ref_psd_rms', self.ref_mic_psd)
        self._create_array('exp_psd_rms', self.exp_mic_psd)
        self._create_array('exp_mic_sens', self.exp_mic_sens)
        self._create_array('speaker_spl', self.speaker_spl)
        self._create_array('time', self.time)
        self._create_array('waveform', self.waveform)


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component, None, {'bgcolor': 'transparent'})

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=40,
                                   bgcolor='transparent')

        sig_plot = create_line_plot((self.data.time, self.data.waveform),
                                    color='black', bgcolor='white')
        axis = PlotAxis(component=sig_plot, orientation='left',
                        title="Cal. sig. (V)")
        sig_plot.underlays.append(axis)
        container.insert(0, sig_plot)

        # Overlay the experiment and reference microphone signal
        overlay = OverlayPlotContainer()
        time = self.data.ref_microphone.time
        signal = self.data.ref_microphone.get_average()*1e3
        ref_plot = create_line_plot((time, signal), color='black')
        ref_plot.alpha = 0.5
        axis = PlotAxis(component=ref_plot, orientation='left',
                        title="Ref. mic. signal (mV)")
        ref_plot.underlays.append(axis)
        overlay.insert(0, ref_plot)
        time = self.data.exp_microphone.time
        signal = self.data.exp_microphone.get_average()*1e3
        exp_plot = create_line_plot((time, signal), color='red')
        axis = PlotAxis(component=exp_plot, orientation='right',
                        title="Exp. mic. signal (mV)")
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title="Time (msec)")
        exp_plot.underlays.append(axis)
        zoom = BetterSelectingZoom(component=exp_plot, tool_mode='range',
                                   always_on=True, axis='index')
        exp_plot.underlays.append(zoom)
        pan = PanTool(component=exp_plot, axis='index')
        exp_plot.tools.append(pan)

        ref_plot.index_mapper = exp_plot.index_mapper
        sig_plot.index_mapper = exp_plot.index_mapper
        axis = PlotAxis(component=sig_plot, orientation='bottom',
                        title="Time (msec)")
        sig_plot.underlays.append(axis)

        overlay.insert(0, exp_plot)
        container.insert(0, overlay)

        frequency = self.data.frequency[1:]
        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        # Overlay the experiment and reference microphone response (FFT)
        exp_mic_db = self.data.exp_mic_psd[1:]
        ref_mic_db = self.data.ref_mic_psd[1:]

        ref_plot = create_line_plot((frequency, ref_mic_db), color='black')
        ref_plot.alpha = 0.5
        exp_plot = create_line_plot((frequency, exp_mic_db), color='red')
        ref_plot.index_mapper = index_mapper
        exp_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title='Frequency (Hz)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='left',
                        title='Mic. PSD (dB re V)')
        exp_plot.underlays.append(axis)
        zoom = BetterSelectingZoom(component=exp_plot, tool_mode='range',
                                   always_on=True, axis='index')
        exp_plot.underlays.append(zoom)
        pan = PanTool(component=exp_plot, axis='index')
        exp_plot.tools.append(pan)
        ref_plot.value_mapper = exp_plot.value_mapper

        overlay = OverlayPlotContainer(ref_plot, exp_plot)
        container.insert(0, overlay)

        # Convert to dB re mV
        exp_mic_sens_db = self.data.exp_mic_sens[1:]
        plot = create_line_plot((frequency, exp_mic_sens_db), color='red',
                                bgcolor='white')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. sens. V (dB re Pa)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        # Plot the speaker output
        speaker_spl = self.data.speaker_spl[1:]
        spl_plot = create_line_plot((frequency, speaker_spl), color='black',
                                    bgcolor='white')
        spl_plot.index_mapper = index_mapper
        axis = PlotAxis(component=spl_plot, orientation='left',
                        title="Speaker output (dB SPL)")
        spl_plot.underlays.append(axis)
        axis = PlotAxis(component=spl_plot, orientation='bottom',
                        title="Frequency (Hz)")
        spl_plot.underlays.append(axis)

        sig_psd = self.data.sig_psd[1:]
        sig_frequency = self.data.sig_frequency[1:]
        sig_plot = create_line_plot((sig_frequency, sig_psd), color='blue',
                                    bgcolor='white')
        sig_plot.index_mapper = index_mapper

        overlay = OverlayPlotContainer(spl_plot, sig_plot)
        container.insert(0, overlay)

        self.container = container

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                VGroup(
                    Item('container', editor=ComponentEditor(), width=500,
                         height=800, show_label=False),
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


class EarChirpCal(HasTraits):

    paradigm = Instance(EarChirpCalSettings, ())
    data = Instance(EarChirpCalData)
    container = Instance(Component, None, {'bgcolor': 'transparent'})

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=40,
                                   bgcolor='transparent')

        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        pd = ArrayPlotData(
            frequency=self.data.frequency[1:],
            speaker_spl=self.data.speaker_spl[1:],
            speaker_spl_norm=self.data.speaker_spl_norm[1:],
            exp_mic_psd=self.data.exp_mic_psd[1:],
            time=self.data.time,
            signal=self.data.waveform,
            exp_mic=self.data.exp_microphone.get_average()*1e3,
        )

        speaker_plot = Plot(pd, bgcolor='white', padding=0)
        speaker_plot.plot(('frequency', 'speaker_spl'), color='black')
        speaker_plot.plot(('frequency', 'speaker_spl_norm'), color='gray')
        container.add(speaker_plot)

        mic_psd_plot = Plot(pd, bgcolor='white', padding=0)
        mic_psd_plot.plot(('frequency', 'exp_mic_psd'), color='black')
        container.add(mic_psd_plot)

        mic_plot = Plot(pd, bgcolor='white', padding=0)
        mic_plot.plot(('time', 'exp_mic'), color='black')
        container.add(mic_plot)

        sig_plot = Plot(pd, bgcolor='white', padding=0)
        sig_plot.plot(('time', 'signal'), color='black')
        container.add(sig_plot)

        self.container = container

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                VGroup(
                    Item('container', editor=ComponentEditor(), width=500,
                         height=800, show_label=False),
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


class BaseChirpCalController(AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)
    complete = Bool(False)
    fh = Any(None)
    filename = Any

    def get_waveform_chirp(self):
        # Load the variables
        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        epoch_duration = self.get_current_value('duration')
        vrms = self.get_current_value('amplitude')
        rise_time = self.get_current_value('rise_time')
        start_atten = self.get_current_value('start_attenuation')
        end_atten = self.get_current_value('end_attenuation')
        calibration = get_chirp_transform(vrms, start_atten, end_atten)

        # By using an Attenuation calibration and setting tone level to 0, a
        # sine wave at the given amplitude (as specified in the settings) will
        # be generated at each frequency as the reference.
        ramp = blocks.LinearRamp(name='sweep')
        token = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope')

        token.set_value('sweep.ramp_duration', epoch_duration)
        token.set_value('envelope.duration', epoch_duration)
        token.set_value('envelope.rise_time', rise_time)
        token.set_value('sweep.start', freq_lb)
        token.set_value('sweep.stop', freq_ub)

        return generate_waveform(token, self.dac_fs, duration=epoch_duration,
                                 calibration=calibration, vrms=vrms)

    def get_waveform(self):
        a, b = golay_pair(16)
        return a

    def save(self, info=None):
        filename = get_save_file(settings.CALIBRATION_DIR,
                                 'Microphone calibration|*.mic')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.model.data.fh.flush()
            shutil.copy(self.filename, filename)

    def start(self, info=None):
        log.debug('Starting calibration')
        try:
            self.complete = False
            self.epochs_acquired = 0
            self.complete = False
            self.calibration_accepted = False
            if self.fh is not None:
                self.fh.close()
            self.initialize_context()
            self.refresh_context()
            self.state = 'running'
            self._setup_input()
            self._setup_output()
            self.waveforms = []
            self.iface_dac.start()
        except Exception as e:
            raise
            if info is not None:
                error(info.ui.control, str(e))
            else:
                error(None, str(e))
            self.stop(info)

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.complete = True

    def _setup_input(self):
        log.debug('Setting up input')
        epoch_duration = self.get_current_value('duration')
        exp_range = self.get_current_value('exp_range')
        self.iface_adc = ni.TriggeredDAQmxSource(
            fs=self.adc_fs, epoch_duration=epoch_duration,
            input_line=self.MIC_INPUT, callback=self.poll,
            trigger_duration=10e-3, expected_range=exp_range)
        fs = self.adc_fs
        self.model.data._create_microphone_node(fs, epoch_duration, 'ref')
        self.model.data._create_microphone_node(fs, epoch_duration, 'exp')
        self.iface_adc.setup()
        self.iface_adc.start()

    def _setup_output(self):
        iti = self.get_current_value('iti')
        output = self.get_current_value('output')
        averages = self.get_current_value('averages')
        output_gain = self.get_current_value('output_gain')
        analog_output = '/{}/{}'.format(ni.DAQmxDefaults.DEV, output)

        self.waveform = self.get_waveform()
        duration = self.waveform.shape[-1]/self.dac_fs
        total_duration = (duration+iti)*averages
        self.iface_dac = ni.DAQmxPlayer(fs=self.dac_fs,
                                        total_duration=total_duration,
                                        allow_regen=True)
        self.iface_dac.setup()
        self.iface_dac.write(*prepare_for_write(self.waveform, self.dac_fs,
                                                iti))

        # Configure the attenuation
        self.iface_att = ni.DAQmxAttenControl()
        self.iface_att.setup()
        if output == 'ao0':
            self.iface_att.set_gain(-np.inf, output_gain)
        elif output == 'ao1':
            self.iface_att.set_gain(output_gain, -np.inf)

    def run_reference_calibration(self, info=None):
        standard.launch_gui(parent=info.ui.control, kind='livemodal')

    def save_settings(self, info=None):
        self.save_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')

    def load_settings(self, info=None):
        self.load_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')


class EarChirpCalController(BaseChirpCalController):

    MIC_INPUT = ni.DAQmxDefaults.MIC_INPUT

    def finalize(self):
        # This is the chirp waveform
        waveform_fs = self.iface_dac.fs
        exp_mic_gain = self.get_current_value('exp_mic_gain')
        waveform_averages = self.get_current_value('waveform_averages')
        self.model.data.compute_transfer_functions(waveform_fs,
                                                   self.waveform,
                                                   exp_mic_gain,
                                                   waveform_averages,
                                                   self.calibration)
        self.model.data.save(**dict(self.model.paradigm.items()))
        self.model.generate_plots()
        self.complete = True
        self.stop()

    def poll(self, waveform):
        log.debug('Polling')
        self.model.data.exp_microphone.send(waveform[:, 0, :])
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            self.finalize()


class ChirpCalController(BaseChirpCalController):

    MIC_INPUT = '{}, {}'.format(ni.DAQmxDefaults.MIC_INPUT,
                                ni.DAQmxDefaults.REF_MIC_INPUT)

    def finalize(self):
        # This is the chirp waveform
        #waveform = self.iface_dac.realize().ravel()
        waveform_fs = self.iface_dac.fs
        ref_mic_sens = self.get_current_value('ref_mic_sens')
        ref_mic_gain = self.get_current_value('ref_mic_gain')
        exp_mic_gain = self.get_current_value('exp_mic_gain')
        waveform_averages = self.get_current_value('waveform_averages')
        self.model.data.compute_transfer_functions(waveform_fs,
                                                   self.waveform,
                                                   ref_mic_sens,
                                                   ref_mic_gain,
                                                   exp_mic_gain,
                                                   waveform_averages)
        self.model.data.save(**dict(self.model.paradigm.items()))
        self.model.generate_plots()
        self.complete = True
        self.stop()

    def poll(self, waveform):
        log.debug('Polling')
        self.model.data.exp_microphone.send(waveform[:, 0, :])
        self.model.data.ref_microphone.send(waveform[:, 1, :])
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            self.finalize()


def launch_gui(output='ao0', **kwargs):
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    with tables.open_file(tempfile, 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController(filename=tempfile)
        paradigm = ChirpCalSettings(output=output)
        ChirpCal(data=data, paradigm=paradigm) \
            .edit_traits(handler=controller, **kwargs)


def main_chirp_cal():
    import logging
    logging.basicConfig(level='DEBUG')
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    from cochlear import configure_logging
    with tables.open_file(tempfile, 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController(filename=tempfile)
        paradigm = ChirpCalSettings(output='ao0')
        ChirpCal(data=data, paradigm=paradigm) \
            .configure_traits(handler=controller)


def main_ear_cal():
    import logging
    logging.basicConfig(level='DEBUG')
    from neurogen.calibration import InterpCalibration
    import os.path
    mic_file = os.path.join('c:/data/cochlear/calibration',
                            '150407 - calibration with 377C10.mic')
    c = InterpCalibration.from_mic_file(mic_file)

    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    from cochlear import configure_logging
    with tables.open_file(tempfile, 'w') as fh:
        data = EarChirpCalData(store_node=fh.root)
        controller = EarChirpCalController(filename=tempfile, calibration=c)
        paradigm = EarChirpCalSettings(output='ao0')
        EarChirpCal(data=data, paradigm=paradigm) \
            .configure_traits(handler=controller)

if __name__ == '__main__':
    #main_ear_cal()
    main_chirp_cal()
