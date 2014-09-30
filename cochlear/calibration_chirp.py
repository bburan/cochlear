from __future__ import division

import numpy as np

from traits.api import (HasTraits, Float, Instance, Int, on_trait_change,
                        Property)
from traitsui.api import View, Item, HGroup, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, LogMapper, LinearMapper, VPlotContainer,
                       PlotAxis)
#from chaco.tools.api import BetterZoom, PanTool

from experiment import icon_dir, AbstractController, AbstractData
from experiment.channel import FileEpochChannel
from experiment.plots.fft_channel_plot import FFTChannelPlot
from experiment.plots.epoch_channel_plot import EpochChannelPlot
from experiment.plots.helpers import add_default_grids, add_time_axis

from pyface.timer.api import Timer

from neurogen import block_definitions as blocks
from neurogen.calibration import Attenuation

from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)

DAC_FS = 200e3
ADC_FS = 200e3


class ChirpCalSettings(HasTraits):

    kw = dict(context=True)
    mic_sens = Float(0.015, label='Mic. sens. (Vrms/Pa)', **kw)
    freq_lb = Float(0.5e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(40e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    averages = Int(24, label='Number of averages', **kw)
    ici = Float(0.01, label='Inter-chirp interval', **kw)
    delay = Float(0.01, label='Chirp delay (sec)', **kw)

    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, ici, delay',
                              label='Total duration (sec)')

    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.delay+self.duration+self.ici)*self.averages

    traits_view = View(
        VGroup(
            'mic_sens',
            'freq_lb',
            'freq_ub',
            'freq_resolution',
            'averages',
            'ici',
            'delay',
            Item('duration', style='readonly'),
            Item('total_duration', style='readonly'),
            show_border=True,
            label='Chirp settings',
        )
    )


class ChirpCalData(AbstractData):

    microphone = Instance('experiment.channel.EpochChannel')

    def _create_microphone_node(self, fs, epoch_duration):
        fh = self.store_node._v_file
        epoch_n = int(fs*epoch_duration)
        node = FileEpochChannel(node=fh.root, name='microphone',
                                epoch_size=epoch_n, fs=fs, dtype=np.double,
                                use_checksum=True)
        self.microphone = node


class ChirpCalController(DAQmxDefaults, AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)

    def start(self, info=None):
        self.state = 'running'
        self.initialize_context()
        self.refresh_context()

        duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        delay = self.get_current_value('delay')
        #epoch_duration = duration+ici+delay
        epoch_duration = duration+ici
        averages = self.get_current_value('averages')

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.MIC_INPUT,
                                              counter_line=self.MIC_COUNTER,
                                              trigger_line=self.MIC_TRIGGER,
                                              callback=self.poll)

        self.iface_dac = DAQmxSink(name='sink', fs=self.dac_fs,
                                   calibration=Attenuation(vrms=1),
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
        self.model.data._create_microphone_node(self.adc_fs, epoch_duration)

        #self.current_graph.set_value('delay.delay', delay)

        freq_lb = self.get_current_value('freq_lb')
        self.current_graph.set_value('sweep.start', freq_lb)

        freq_ub = self.get_current_value('freq_ub')
        self.current_graph.set_value('sweep.stop', freq_ub)

        self.current_graph.queue_init('FIFO')
        self.current_graph.queue_append(averages)

        # Initialize and reserve the NIDAQmx hardware
        self.iface_adc.setup()
        self.iface_att.setup()

        # We want to generate the calibration at unity gain.
        self.iface_att.set_mute(False)
        self.iface_att.set_zero_crossing(False)
        self.iface_att.set_gain(0, 0)
        self.iface_att.clear()

        self.iface_adc.start()
        self.current_graph.play_queue()

    def stop(self, info=None):
        self.state = 'uninitialized'
        self.iface_dac.clear()
        self.iface_adc.clear()

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=-1)
        self.model.data.microphone.send(waveform)
        self.epochs_acquired += 2
        if self.epochs_acquired == self.get_current_value('averages'):
            self.model.generate_plots()
            self.stop()


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        container = VPlotContainer(padding=[50, 50, 50, 50], spacing=50)

        # Generate the time-series plot
        duration = self.paradigm.duration+self.paradigm.ici
        index_range = DataRange1D(low_setting=0, high_setting=duration)
        index_mapper = LinearMapper(range=index_range)
        value_range = DataRange1D(low_setting=-2, high_setting=2)
        value_mapper = LinearMapper(range=value_range)
        plot = EpochChannelPlot(index_mapper=index_mapper,
                                value_mapper=value_mapper,
                                source=self.data.microphone)
        add_default_grids(plot, major_index=0.01, minor_index=0.001)
        axis = PlotAxis(component=plot, orientation='bottom', title="Time")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='left', title="Signal (V)")
        plot.underlays.append(axis)
        #zoom = BetterZoom(plot)
        #plot.overlays.append(zoom)
        #pan = PanTool(plot)
        #plot.overlays.append(pan)
        container.add(plot)

        index_range = DataRange1D(low_setting=0, high_setting=100e3)
        index_mapper = LinearMapper(range=index_range)
        value_range = DataRange1D(low_setting=0, high_setting=1e7)
        value_mapper = LogMapper(range=value_range)
        plot = FFTChannelPlot(index_mapper=index_mapper,
                              value_mapper=value_mapper,
                              source=self.data.microphone)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='left', title="Specturm (V)")
        plot.underlays.append(axis)
        container.add(plot)
        self.container = container

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                VGroup(
                    Item('container', editor=ComponentEditor(), width=500,
                        height=600, show_label=False),
                ),
            ),
            show_labels=False,
        ),
        toolbar=ToolBar(
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='Stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.state=="running"'),
        ),
        resizable=True,
    )


if __name__ == '__main__':
    import tables
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController()
        ChirpCal(data=data).configure_traits(handler=controller)
