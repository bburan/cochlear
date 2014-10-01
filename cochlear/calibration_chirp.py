from __future__ import division

import numpy as np

from traits.api import (HasTraits, Float, Instance, Int, Property, Any)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, LinearMapper, VPlotContainer, PlotAxis,
                       create_line_plot)

from experiment import icon_dir, AbstractController, AbstractData
from experiment.channel import FileEpochChannel
from experiment.plots.fft_channel_plot import FFTChannelPlot
from experiment.plots.epoch_channel_plot import EpochChannelPlot

from neurogen import block_definitions as blocks
from neurogen.calibration import Attenuation

from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)

DAC_FS = 200e3
ADC_FS = 200e3


class ChirpCalSettings(HasTraits):

    kw = dict(context=True)
    ref_mic_sens = Float(0.015, label='Ref. mic. sens. (V/Pa)', **kw)
    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', **kw)
    exp_mic_gain = Float(40, label='Exp. mic. gain (dB)', **kw)

    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    attenuation = Float(30, label='Attenaution (dB)', **kw)
    freq_lb = Float(0.5e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(40e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    averages = Int(128, label='Number of averages', **kw)
    ici = Float(0.005, label='Inter-chirp interval', **kw)

    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, ici',
                              label='Total duration (sec)')

    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.duration+self.ici)*self.averages

    traits_view = View(
        VGroup(
            VGroup(
                'ref_mic_sens',
                'ref_mic_gain',
                'exp_mic_gain',
                label='Hardware settings',
                show_border=True,
            ),
            VGroup(
                'freq_lb',
                'freq_ub',
                'freq_resolution',
                'averages',
                'ici',
                'attenuation',
                'amplitude',
                Item('duration', style='readonly'),
                Item('total_duration', style='readonly'),
                show_border=True,
                label='Chirp settings',
            ),
        )
    )


class ChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    ref_microphone = Instance('experiment.channel.EpochChannel')
    ref_mic_sens = Float()

    speaker_tf = Any()
    exp_mic_tf = Any()

    def _create_microphone_nodes(self, fs, epoch_duration):
        fh = self.store_node._v_file
        epoch_n = int(fs*epoch_duration)
        node = FileEpochChannel(node=fh.root, name='exp_microphone',
                                epoch_size=epoch_n, fs=fs, dtype=np.double,
                                use_checksum=True)
        self.exp_microphone = node
        node = FileEpochChannel(node=fh.root, name='ref_microphone',
                                epoch_size=epoch_n, fs=fs, dtype=np.double,
                                use_checksum=True)
        self.ref_microphone = node

    def compute_transfer_functions(self, ref_mic_sens):
        ref_psd = self.ref_microphone.get_average_psd()
        exp_psd = self.exp_microphone.get_average_psd()
        self.speaker_tf = ref_psd/ref_mic_sens
        self.exp_mic_tf = exp_psd/self.speaker_tf
        self.frequency = self.ref_microphone.get_fftfreq()

        fh = self.store_node._v_file
        data = np.c_[self.frequency, self.speaker_tf]
        s_node = fh.create_array(fh.root, 'speaker_tf', data)
        s_node._v_attrs['ref_mic_sens'] = ref_mic_sens
        data = np.c_[self.frequency, self.exp_mic_tf]
        fh.create_array(fh.root, 'exp_mic_tf', data)


class ChirpCalController(DAQmxDefaults, AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)

    MIC_INPUT = '/{}/ai0:1'.format(DAQmxDefaults.DEV)

    def start(self, info=None):
        self.state = 'running'
        self.initialize_context()
        self.refresh_context()

        duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        epoch_duration = duration
        averages = self.get_current_value('averages')

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.MIC_INPUT,
                                              counter_line=self.MIC_COUNTER,
                                              trigger_line=self.MIC_TRIGGER,
                                              callback=self.poll,
                                              trigger_duration=10e-3)

        calibration = Attenuation(vrms=self.get_current_value('amplitude'))
        self.iface_dac = DAQmxSink(name='sink',
                                   fs=self.dac_fs,
                                   calibration=calibration,
                                   output_line=self.SPEAKER_OUTPUT,
                                   trigger_line=self.SPEAKER_TRIGGER,
                                   run_line=self.SPEAKER_RUN)

        self.iface_att = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                           cs_line=self.VOLUME_CS,
                                           data_line=self.VOLUME_SDI,
                                           mute_line=self.VOLUME_MUTE,
                                           zc_line=self.VOLUME_ZC,
                                           hw_clock=self.DIO_CLOCK)

        # By using an Attenuation calibration and setting tone level to 0, a 1
        # Vrms sine wave will be generated at each frequency as the reference.
        # The chirp will be bounded in a cosine-squared envelope, so we want to
        # ensure that the frequency sweep doesn't begin until the envelope
        # rise-time is over.
        ramp = blocks.LinearRamp(name='sweep')
        graph = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope', rise_time=2.5e-3) >> \
            self.iface_dac
        self.current_graph = graph

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

        # We want to generate the calibration at unity gain.
        attenuation = self.get_current_value('attenuation')
        self.iface_att.set_mute(False)
        self.iface_att.set_zero_crossing(False)
        self.iface_att.set_gain(-attenuation, -attenuation)
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
            self.model.data.compute_transfer_functions(ref_mic_sens)
            self.model.generate_plots()
            self.stop()


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        self.container = VPlotContainer(padding=[70, 20, 20, 50], spacing=50)

        duration = self.paradigm.duration+self.paradigm.ici
        index_range = DataRange1D(low_setting=0, high_setting=duration)
        index_mapper = LinearMapper(range=index_range)
        value_range = DataRange1D(low_setting=-2, high_setting=2)
        value_mapper = LinearMapper(range=value_range)
        plot = EpochChannelPlot(index_mapper=index_mapper,
                                value_mapper=value_mapper,
                                source=self.data.ref_microphone)
        axis = PlotAxis(component=plot, orientation='left',
                        title="Ref. mic. signal (V)")
        plot.underlays.append(axis)
        self.container.insert(0, plot)

        value_mapper = LinearMapper(range=value_range)
        plot = EpochChannelPlot(index_mapper=index_mapper,
                                value_mapper=value_mapper,
                                source=self.data.exp_microphone)
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. signal (V)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom', title="Time")
        plot.underlays.append(axis)
        self.container.insert(0, plot)

        index_range = DataRange1D(low_setting=self.paradigm.freq_lb-1000,
                                  high_setting=self.paradigm.freq_ub+1000)
        index_mapper = LinearMapper(range=index_range)
        value_range = DataRange1D(low_setting=0, high_setting=140)
        value_mapper = LinearMapper(range=value_range)

        # Convert microphone sensitivity to a reference value that can be used
        # to compute dB SPL in the FFTChannelPlot
        db_reference = self.paradigm.ref_mic_sens*20e-6
        plot = FFTChannelPlot(index_mapper=index_mapper,
                              value_mapper=value_mapper,
                              source=self.data.ref_microphone,
                              db=True,
                              reference=db_reference)
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker output (dB SPL)")
        plot.underlays.append(axis)
        self.container.insert(0, plot)

        plot = create_line_plot((self.data.frequency, self.data.exp_mic_tf*1e3))
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. sens (mV/Pa)")
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
        ),
        resizable=True,
    )


def launch_gui(**kwargs):
    import tables
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController()
        ChirpCal(data=data).edit_traits(handler=controller, **kwargs)


def main():
    import tables
    import logging
    logging.basicConfig(level='DEBUG')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController()
        ChirpCal(data=data).configure_traits(handler=controller)


if __name__ == '__main__':
    main()
