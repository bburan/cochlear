from __future__ import division

import tempfile
import os.path
import shutil

import numpy as np
import tables

from traits.api import (HasTraits, Float, Instance, Int, Property, Any, Bool,
                        Enum)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit, Include
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment import (icon_dir, AbstractController, AbstractData,
                        AbstractParadigm)
from experiment.channel import FileFilteredEpochChannel
from experiment.util import get_save_file

from neurogen import block_definitions as blocks
from neurogen.calibration import SimpleCalibration
from neurogen.util import patodb, db

from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)

DAC_FS = 200e3
ADC_FS = 200e3


class ChirpCalSettings(AbstractParadigm):

    kw = dict(context=True)
    exp_mic_gain = Float(40, label='Exp. mic. gain (dB)', **kw)

    output = Enum(('ao0', 'ao1'), label='Analog output (channel)', **kw)
    rise_time = Float(2.5e-3, label='Envelope rise time', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    output_gain = Float(-20, label='Output gain (dB)', **kw)
    freq_lb = Float(0.4e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(40e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(5, label='Frequency resolution (Hz)')
    fft_averages = Int(1, label='Number of FFTs', **kw)
    waveform_averages = Int(10, label='Number of chirps per FFT', **kw)
    ici = Float(0.01, label='Inter-chirp interval', **kw)

    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Number of chirps', **kw)
    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, ici',
                              label='Total duration (sec)')

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

    mic_settings = VGroup(
        'exp_mic_gain',
        label='Microphone settings',
        show_border=True,
    )

    traits_view = View(
        VGroup(
            Include('output_settings'),
            Include('mic_settings'),
            Include('stimulus_settings'),
        )
    )


class ReferenceChirpCalSettings(ChirpCalSettings):

    kw = dict(context=True)
    ref_mic_sens = Float(2.0e-3, label='Ref. mic. sens. (V/Pa)', **kw)
    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', **kw)

    mic_settings = VGroup(
        'ref_mic_sens',
        'ref_mic_gain',
        'exp_mic_gain',
        label='Microphone settings',
        show_border=True,
    )


class ChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    ref_microphone = Instance('experiment.channel.EpochChannel')
    ref_mic_sens = Float()

    exp_mic_tf = Any()

    @classmethod
    def from_node(cls, node):
        obj = ChirpCalData()
        exp_node = node.exp_microphone
        ref_node = node.ref_microphone
        obj.exp_microphone = FileFilteredEpochChannel.from_node(exp_node)
        obj.ref_microphone = FileFilteredEpochChannel.from_node(ref_node)
        obj.store_node = node
        return obj

    def _create_microphone_nodes(self, fs, epoch_duration):
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

    def compute_transfer_functions(self, ref_mic_sens, ref_mic_gain,
                                   exp_mic_gain, waveform_averages):

        # All functions are computed using these frequencies
        frequency = self.ref_microphone.get_fftfreq()

        # Compute the PSD of each microphone in Vrms
        ref_psd = self.ref_microphone \
            .get_average_psd(waveform_averages=waveform_averages)/np.sqrt(2)
        exp_psd = self.exp_microphone \
            .get_average_psd(waveform_averages=waveform_averages)/np.sqrt(2)

        # Compensate for measurement gain settings
        ref_psd = ref_psd/(10**(ref_mic_gain/20.0))
        exp_psd = exp_psd/(10**(exp_mic_gain/20.0))

        # Actual output of speaker in pascals
        speaker_pa = ref_psd/ref_mic_sens
        speaker_spl = patodb(speaker_pa)

        # Sensitivity of experiment microphone as function of frequency
        # (Vrms/Pa)
        exp_mic_sens = exp_psd/speaker_pa

        self._create_array('frequency', frequency)
        self._create_array('ref_psd_vrms', ref_psd)
        self._create_array('exp_psd_vrms', exp_psd)
        self._create_array('speaker_spl', speaker_spl)
        self._create_array('exp_mic_sens', exp_mic_sens)


class ChirpCalController(DAQmxDefaults, AbstractController):

    filename = os.path.join(tempfile.gettempdir(),
                            'microphone_calibration.hdf5')
    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)
    complete = Bool(False)
    fh = Any(None)

    MIC_INPUT = '{}, {}'.format(DAQmxDefaults.MIC_INPUT,
                                DAQmxDefaults.REF_MIC_INPUT)

    def save(self, info=None):
        filename = get_save_file('c:/', 'Microphone calibration|*_miccal.hdf5')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.fh.flush()
            shutil.copy(self.filename, filename)

    def start(self, info=None):
        self.complete = False
        self.state = 'running'
        if self.fh is not None:
            self.fh.close()
        self.fh = tables.open_file(self.filename, 'w')
        self.model.data = ChirpCalData(store_node=self.fh.root)
        self.initialize_context()
        self.refresh_context()

        rise_time = self.get_current_value('rise_time')
        output = self.get_current_value('output')
        duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        epoch_duration = duration
        averages = self.get_current_value('averages')
        output_gain = self.get_current_value('output_gain')
        vrms = self.get_current_value('amplitude')
        calibration = SimpleCalibration.as_attenuation(vrms=vrms)

        analog_output = '/{}/{}'.format(self.DEV, output)

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.MIC_INPUT,
                                              counter_line=self.AI_COUNTER,
                                              trigger_line=self.AI_TRIGGER,
                                              callback=self.poll,
                                              trigger_duration=10e-3)

        self.iface_att = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                           cs_line=self.VOLUME_CS,
                                           data_line=self.VOLUME_SDI,
                                           mute_line=self.VOLUME_MUTE,
                                           zc_line=self.VOLUME_ZC,
                                           hw_clock=self.DIO_CLOCK)
        self.iface_dac = DAQmxSink(name='sink',
                                   fs=self.dac_fs,
                                   calibration=calibration,
                                   output_line=analog_output,
                                   trigger_line=self.SPEAKER_TRIGGER,
                                   run_line=self.SPEAKER_RUN,
                                   attenuator=self.iface_att,
                                   fixed_attenuation=True,
                                   hw_attenuation=0)

        # By using an Attenuation calibration and setting tone level to 0, a
        # sine wave at the given amplitude (as specified in the settings) will
        # be generated at each frequency as the reference.
        ramp = blocks.LinearRamp(name='sweep')
        graph = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope') >> \
            self.iface_dac
        self.current_graph = graph

        self.current_graph.set_value('envelope.rise_time', rise_time)
        self.current_graph.set_value('sweep.ramp_duration', duration)
        self.current_graph.set_value('envelope.duration', duration)
        self.current_graph.set_value('sink.duration', epoch_duration)
        self.model.data._create_microphone_nodes(self.adc_fs, epoch_duration)

        freq_lb = self.get_current_value('freq_lb')
        self.current_graph.set_value('sweep.start', freq_lb)

        freq_ub = self.get_current_value('freq_ub')
        self.current_graph.set_value('sweep.stop', freq_ub)

        self.current_graph.queue_init('FIFO')
        self.current_graph.queue_append(averages, ici)

        # Initialize and reserve the NIDAQmx hardware
        self.iface_adc.setup()
        self.iface_att.setup()

        self.iface_att.set_mute(False)
        self.iface_att.set_zero_crossing(False)
        if output == 'ao0':
            self.iface_att.set_gain(-np.inf, output_gain)
        elif output == 'ao1':
            self.iface_att.set_gain(output_gain, -np.inf)

        # Need to clear this so that the signal output can take over.
        self.iface_att.clear()
        self.iface_adc.start()
        self.current_graph.play_queue()

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_dac.clear()
        self.iface_adc.clear()

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.exp_microphone.send(waveform[0])
        self.model.data.ref_microphone.send(waveform[1])
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            ref_mic_sens = self.get_current_value('ref_mic_sens')
            ref_mic_gain = self.get_current_value('ref_mic_gain')
            exp_mic_gain = self.get_current_value('exp_mic_gain')
            waveform_averages = self.get_current_value('waveform_averages')
            self.model.data.compute_transfer_functions(ref_mic_sens,
                                                       ref_mic_gain,
                                                       exp_mic_gain,
                                                       waveform_averages)
            self.model.data.save(**dict(self.model.paradigm.items()))
            self.model.generate_plots()
            self.complete = True
            self.stop()


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        self.container = VPlotContainer(padding=70, spacing=70)

        # Overlay the experiment and reference microphone signal
        overlay = OverlayPlotContainer()
        time = self.data.ref_microphone.time
        signal = self.data.ref_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
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
        axis = PlotAxis(component=plot, orientation='bottom', title="Time")
        plot.underlays.append(axis)
        overlay.insert(0, plot)
        self.container.insert(0, overlay)

        averages = self.paradigm.waveform_averages

        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        # Overlay the experiment and reference microphone response (FFT)
        frequency = self.data.ref_microphone.get_fftfreq()
        ref_psd_vrms = self.data.ref_microphone \
            .get_average_psd(waveform_averages=averages)/np.sqrt(2)
        exp_psd_vrms = self.data.exp_microphone \
            .get_average_psd(waveform_averages=averages)/np.sqrt(2)
        ref_plot = create_line_plot((frequency[1:], db(ref_psd_vrms[1:], 1e-3)),
                                    color='black')
        exp_plot = create_line_plot((frequency[1:], db(exp_psd_vrms[1:], 1e-3)),
                                    color='red')
        ref_plot.index_mapper = index_mapper
        exp_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title='Frequency (Hz)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='right',
                        title='Exp. mic. resp (dB re 1mV)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=ref_plot, orientation='left',
                        title='Ref. mic. resp (dB re 1mV)')
        ref_plot.underlays.append(axis)
        overlay = OverlayPlotContainer(ref_plot, exp_plot)
        self.container.insert(0, overlay)

        # Convert the refernece microphone response to speaker output
        ref_psd_pa = ref_psd_vrms/self.paradigm.ref_mic_sens
        ref_psd_spl = patodb(ref_psd_pa)
        plot = create_line_plot((frequency[1:], ref_psd_spl[1:]), color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker output (dB SPL)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        self.container.insert(0, plot)

        plot = create_line_plot((self.data.frequency,
                                 db(self.data.exp_mic_tf, 1e-3)))
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. sens mV (dB re Pa)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)

        self.container.insert(0, plot)

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
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            Action(name='Save', action='save',
                   image=ImageResource('document_save', icon_dir),
                   enabled_when='handler.complete')
        ),
        resizable=True,
    )


def launch_gui(**kwargs):
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController()
        ChirpCal(data=data).edit_traits(handler=controller, **kwargs)


def main():
    paradigm = ReferenceChirpCalSettings()
    controller = ChirpCalController()
    ChirpCal(paradigm=paradigm).configure_traits(handler=controller)


if __name__ == '__main__':
    import PyDAQmx as ni
    ni.DAQmxResetDevice('Dev1')
    main()
