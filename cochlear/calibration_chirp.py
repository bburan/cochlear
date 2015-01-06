from __future__ import division

import shutil

import numpy as np

from traits.api import (Float, Int, Property, Any, Bool, Enum)
from traitsui.api import View, Item, VGroup, Include
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment import (icon_dir, AbstractController, AbstractParadigm)
from experiment.util import get_save_file

from neurogen import block_definitions as blocks
from neurogen.calibration import InterpCalibration
from neurogen.util import patodb, db

import os.path
import shutil

import numpy as np
import tables

from traits.api import (HasTraits, Float, Instance, Any, Property)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment import (icon_dir, AbstractData)
from experiment.channel import FileFilteredEpochChannel
from experiment.util import get_save_file

from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear import calibration_standard as reference_calibration
import settings

DAC_FS = 200e3
ADC_FS = 200e3


def get_chirp_transform(vrms):
    calibration_data = np.array([
        (1, 30),
        (100e3, -10),
    ])
    frequencies = calibration_data[:, 0]
    magnitude = calibration_data[:, 1]
    return InterpCalibration.from_single_vrms(frequencies, magnitude, vrms)


class ChirpCalSettings(AbstractParadigm):

    kw = dict(context=True)

    output = Enum(('ao1', 'ao0'), label='Analog output (channel)', **kw)
    rise_time = Float(5e-3, label='Envelope rise time', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    output_gain = Float(6, label='Output gain (dB)', **kw)
    freq_lb = Float(0.1e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(60e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    fft_averages = Int(32, label='Number of FFTs', **kw)
    waveform_averages = Int(32, label='Number of chirps per FFT', **kw)
    ici = Float(0.01, label='Inter-chirp interval', **kw)

    ref_mic_sens_mv = Float(2.703, label='Ref. mic. sens. (mV/Pa)', **kw)
    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', **kw)
    ref_mic_sens = Property(depends_on='ref_mic_sens_mv', **kw)
    exp_mic_gain = Float(40, label='Exp. mic. gain (dB)', **kw)

    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Number of chirps', **kw)
    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, ici',
                              label='Total duration (sec)')

    def _get_ref_mic_sens(self):
        return self.ref_mic_sens_mv*1e-3
    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.duration+self.ici)*self.averages

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
        'ref_mic_sens_mv',
        'ref_mic_gain',
        'exp_mic_gain',
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
        'ici',
        Item('averages', style='readonly'),
        Item('duration', style='readonly'),
        Item('total_duration', style='readonly'),
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


class ChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    ref_microphone = Instance('experiment.channel.EpochChannel')
    ref_mic_sens = Float()
    exp_mic_sens = Any()

    @classmethod
    def from_node(cls, node):
        obj = ReferenceChirpCalData()
        exp_node = node.exp_microphone
        ref_node = node.ref_microphone
        obj.exp_microphone = FileFilteredEpochChannel.from_node(exp_node)
        obj.ref_microphone = FileFilteredEpochChannel.from_node(ref_node)
        obj.store_node = node
        return obj

    def _create_microphone_nodes(self, fs, epoch_duration):
        if 'exp_microphone' in self.fh.root:
            self.fh.root.exp_microphone.remove()
            self.fh.root.exp_microphone_ts.remove()
            self.fh.root.ref_microphone.remove()
            self.fh.root.ref_microphone_ts.remove()
        fh = self.store_node._v_file
        filter_kw = dict(filter_freq_hp=5, filter_freq_lp=80e3,
                         filter_btype='bandpass', filter_order=1)
        node = FileFilteredEpochChannel(node=fh.root, name='exp_microphone',
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        self.exp_microphone = node
        node = FileFilteredEpochChannel(node=fh.root, name='ref_microphone',
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        self.ref_microphone = node

    def _create_array(self, name, array, store_node=None):
        if store_node is None:
            store_node = self.store_node
        if name in store_node:
            store_node._f_get_child(name).remove()
        return self.fh.create_array(store_node, name, array)

    def compute_transfer_functions(self, waveform_fs, waveform, ref_mic_sens,
                                   ref_mic_gain, exp_mic_gain,
                                   waveform_averages):

        self.time = np.arange(len(waveform))/waveform_fs
        self.waveform = waveform
        print ref_mic_sens, ref_mic_gain, exp_mic_gain

        # All functions are computed using these frequencies
        self.frequency = self.ref_microphone.get_fftfreq()

        from neurogen.calibration.util import psd_freq, psd
        # Compute the PSD of each microphone in Vrms
        self.frequency = psd_freq(self.ref_microphone[:],
                                  self.ref_microphone.fs)

        args = 'boxcar', waveform_averages
        ref_psd = psd(self.ref_microphone[:], self.ref_microphone.fs, *args)
        exp_psd = psd(self.exp_microphone[:], self.exp_microphone.fs, *args)

        # Convert to dB re 1V and compensate for measurement gain settings
        self.ref_mic_psd = db(ref_psd.mean(axis=0))-ref_mic_gain
        self.exp_mic_psd = db(exp_psd.mean(axis=0))-exp_mic_gain
        print self.ref_mic_psd.shape
        print self.exp_mic_psd.shape
        print self.frequency.shape

        # Sensitivity of experiment microphone as function of frequency
        # expressed as Vrms (dB re Pa).  This is equivalent to
        # (Vprobe/Vcal)/Ccal
        self.exp_mic_sens = self.exp_mic_psd+db(ref_mic_sens)-self.ref_mic_psd

        self._create_array('frequency', self.frequency)
        self._create_array('ref_psd_rms', self.ref_mic_psd)
        self._create_array('exp_psd_rms', self.exp_mic_psd)
        self._create_array('exp_mic_sens', self.exp_mic_sens)
        self._create_array('time', self.time)
        self._create_array('waveform', self.waveform)


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)


        plot = create_line_plot((self.data.time, self.data.waveform),
                                color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Cal. sig. (V)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (msec)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        # Overlay the experiment and reference microphone signal
        overlay = OverlayPlotContainer()
        time = self.data.ref_microphone.time
        signal = self.data.ref_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
        plot.alpha = 0.5
        axis = PlotAxis(component=plot, orientation='left',
                        title="Ref. mic. signal (mV)")
        plot.underlays.append(axis)
        overlay.insert(0, plot)
        time = self.data.exp_microphone.time
        signal = self.data.exp_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='red')
        axis = PlotAxis(component=plot, orientation='right',
                        title="Exp. mic. signal (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (msec)")
        plot.underlays.append(axis)
        overlay.insert(0, plot)
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
        axis = PlotAxis(component=exp_plot, orientation='right',
                        title='Exp. mic. (dB re V)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=ref_plot, orientation='left',
                        title='Ref. mic. (dB re V)')
        ref_plot.underlays.append(axis)
        overlay = OverlayPlotContainer(ref_plot, exp_plot)
        container.insert(0, overlay)

        # Convert to dB re mV
        exp_mic_sens_db = self.data.exp_mic_sens[1:]
        plot = create_line_plot((frequency, exp_mic_sens_db), color='red')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="PT sens V (dB re Pa)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)

        container.insert(0, plot)
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
    )


class ChirpCalController(AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)
    complete = Bool(False)
    fh = Any(None)
    filename = Any

    def save(self, info=None):
        filename = get_save_file(settings.CALIBRATION_DIR,
                                 'Microphone calibration|*.mic')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.model.data.fh.flush()
            shutil.copy(self.filename, filename)

    def start(self, info=None):
        self.complete = False
        self.state = 'running'
        self.epochs_acquired = 0
        self.complete = False
        self.calibration_accepted = False
        if self.fh is not None:
            self.fh.close()

        self.initialize_context()
        self.refresh_context()
        self._setup_input()
        self._setup_output()
        self.iface_dac.play_queue()

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.complete = True

    def poll(self, waveform):
        self.model.data.exp_microphone.send(waveform[:, 0, :])
        self.model.data.ref_microphone.send(waveform[:, 1, :])
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            self.finalize()

    def _setup_input(self):
        mic_input = '{}, {}'.format(ni.DAQmxDefaults.MIC_INPUT,
                                    ni.DAQmxDefaults.REF_MIC_INPUT)
        epoch_duration = self.get_current_value('duration')
        self.iface_adc = ni.TriggeredDAQmxSource(
            fs=self.adc_fs, epoch_duration=epoch_duration, input_line=mic_input,
            callback=self.poll, trigger_duration=10e-3, expected_range=1.0)
        self.model.data._create_microphone_nodes(self.adc_fs, epoch_duration)
        self.iface_adc.setup()
        self.iface_adc.start()

    def _setup_output(self, output=None):
        if output is None:
            output = self.get_current_value('output')
        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        epoch_duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        averages = self.get_current_value('averages')
        output_gain = self.get_current_value('output_gain')
        vrms = self.get_current_value('amplitude')
        rise_time = self.get_current_value('rise_time')
        calibration = get_chirp_transform(vrms)
        analog_output = '/{}/{}'.format(ni.DAQmxDefaults.DEV, output)

        self.iface_dac = ni.QueuedDAQmxPlayer(
            fs=self.dac_fs, output_line=analog_output, duration=epoch_duration)

        # By using an Attenuation calibration and setting tone level to 0, a
        # sine wave at the given amplitude (as specified in the settings) will
        # be generated at each frequency as the reference.
        ramp = blocks.LinearRamp(name='sweep')
        self.current_channel = \
            blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope') >> \
            ni.DAQmxChannel(calibration=calibration)
        self.current_channel.set_value('sweep.ramp_duration', epoch_duration)
        self.current_channel.set_value('envelope.duration', epoch_duration)
        self.current_channel.set_value('envelope.rise_time', rise_time)
        self.current_channel.set_value('sweep.start', freq_lb)
        self.current_channel.set_value('sweep.stop', freq_ub)

        self.iface_dac.add_channel(self.current_channel)
        self.iface_dac.queue_init('FIFO')
        self.iface_dac.queue_append(averages, ici)

        self.iface_att = ni.DAQmxAttenControl()
        self.iface_att.setup()
        if output == 'ao0':
            self.iface_att.set_gain(-np.inf, output_gain)
        elif output == 'ao1':
            self.iface_att.set_gain(output_gain, -np.inf)
        self.iface_att.set_mute(False)

    def finalize(self):
        waveform_fs, waveform = \
            self.iface_dac.fs, self.iface_dac.realize().ravel()

        ref_mic_sens = self.get_current_value('ref_mic_sens')
        ref_mic_gain = self.get_current_value('ref_mic_gain')
        exp_mic_gain = self.get_current_value('exp_mic_gain')
        waveform_averages = self.get_current_value('waveform_averages')
        self.model.data.compute_transfer_functions(waveform_fs, waveform,
                                                   ref_mic_sens, ref_mic_gain,
                                                   exp_mic_gain,
                                                   waveform_averages)
        self.model.data.save(**dict(self.model.paradigm.items()))
        self.model.generate_plots()
        self.complete = True
        self.stop()

    def run_reference_calibration(self, info):
        reference_calibration.launch_gui(parent=info.ui.control,
                                         kind='livemodal')


def launch_gui(output='ao0', **kwargs):
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    with tables.open_file(tempfile, 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController(filename=tempfile)
        paradigm = ChirpCalSettings(output=output)
        ChirpCal(data=data, paradigm=paradigm) \
            .edit_traits(handler=controller, **kwargs)


if __name__ == '__main__':
    output = 'ao0'
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    #from cochlear import configure_logging
    #configure_logging('temp.log')
    import logging
    logging.basicConfig(level='DEBUG')
    with tables.open_file(tempfile, 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController(filename=tempfile)
        paradigm = ChirpCalSettings(output=output)
        ChirpCal(data=data, paradigm=paradigm) \
            .configure_traits(handler=controller)
