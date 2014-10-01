from traits.api import Instance, Float, Int, Any, push_exception_handler, Bool
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, HGroup)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import LinearMapper, DataRange1D, PlotAxis
from chaco.tools.api import ZoomTool

import numpy as np

from neurogen.block_definitions import Tone, Cos2Envelope
from neurogen.calibration import Attenuation
from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)


from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment)
from experiment.channel import FileEpochChannel

from experiment.plots.epoch_channel_plot import EpochChannelPlot

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 200e3
ADC_FS = 50e3


class ABRData(AbstractData):

    waveforms = Any
    epoch_size = Int(int(10e-3*ADC_FS))
    waveform_node = Any

    def _waveform_node_default(self):
        fh = self.store_node._v_file
        return fh.create_group(self.store_node, 'waveforms')

    def create_waveform_channel(self, trial):
        # We use a separate "channel" for each trial set.
        channel = FileEpochChannel(node=self.waveform_node,
                                   name='stack_{}'.format(trial),
                                   epoch_size=self.epoch_size, fs=ADC_FS,
                                   dtype=np.double, use_checksum=True)
        self.current_channel = channel
        return channel

    def _waveforms_default(self):
        fh = self.store_node._v_file
        return FileEpochChannel(node=fh.root, name='erp',
                                epoch_size=self.epoch_size, fs=ADC_FS,
                                dtype=np.double, use_checksum=True)


class ABRParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    # Signal acquisition settings
    averages = Expression(20, **kw)
    window = Expression(10e-3, **kw)
    quick_threshold = Expression(550e-9, **kw)
    reject_threshold = Expression(15e-6, **kw)

    # Stimulus settings
    repetition_rate = Expression(10, **kw)
    frequency = Expression(1e3, **kw)
    duration = Expression(4e-3, **kw)
    ramp_duration = Expression(0.5e-3, **kw)
    level = Expression(0, **kw)

    traits_view = View(
        VGroup(
            VGroup(
                'averages',
                'window',
                'quick_threshold',
                label='Acquisition settings',
                show_border=True,
            ),
            VGroup(
                'frequency',
                'duration',
                'ramp_duration',
                'repetition_rate',
                'level',
                label='Stimulus settings',
                show_border=True,
            ),
            VGroup(
                'reject_threshold',
                label='Analysis settings',
                show_border=True,
            ),
        ),
    )


class ABRController(DAQmxDefaults, AbstractController):

    current_time_elapsed = Float(0)
    current_repetitions = Int(0)
    current_valid_repetitions = Int(0)

    iface_adc = Instance('nidaqmx.TriggeredDAQmxSource')
    iface_dac = Instance('nidaqmx.DAQmxSink')

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)

    done = Bool(False)
    current_polarity = Bool(False)
    stop_requested = Bool(False)

    def setup_experiment(self, info=None):
        self.iface_adc.clear()
        self.iface_dac.clear()

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=10e-3,
                                              input_line=self.ERP_INPUT,
                                              counter_line=self.ERP_COUNTER,
                                              trigger_line=self.ERP_TRIGGER,
                                              callback=self.poll)

        self.iface_dac = DAQmxSink(name='sink', fs=self.dac_fs,
                                   calibration=Attenuation(),
                                   output_line=self.SPEAKER_OUTPUT,
                                   trigger_line=self.SPEAKER_TRIGGER,
                                   run_line=self.SPEAKER_RUN)

        self.iface_atten = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                             cs_line=self.VOLUME_CS,
                                             data_line=self.VOLUME_SDI,
                                             mute_line=self.VOLUME_MUTE,
                                             zc_line=self.VOLUME_ZC,
                                             hw_clock=self.DIO_CLOCK)

        self.current_graph = Tone(name='tone') >> \
            Cos2Envelope(name='envelope') >> \
            self.iface_dac

        self.iface_adc.setup()

    def start_experiment(self, info=None):
        self.next_stimulus()

    def stop_experiment(self, info=None):
        self.stop_requested = True

    def next_stimulus(self):
        self.refresh_context(evaluate=True)
        self.model.plot.source = \
            self.model.data.create_waveform_channel(self.current_trial)

        # Set up alternating polarity by shifting the phase np.pi.  Use the
        # Interleaved FIFO queue for this.
        self.current_graph.queue_init('FIFO')
        self.current_graph.set_value('tone.phase', 0)
        self.current_graph.queue_append(np.inf)
        self.current_graph.set_value('tone.phase', np.pi)
        self.current_graph.queue_append(np.inf)

        self.done = False
        self.current_trial += 1
        self.iface_adc.start()
        self.current_graph.play_queue()

    def poll(self):
        # Read in new data and save it.  Even if we've acquired enough
        # averages, we should go ahead and read in any new data we can.
        waveform = self.iface_adc.read_analog(timeout=-1)
        self.model.data.current_channel.send(waveform)
        self.current_repetitions += 1

        # Did the user request a stop?
        if self.stop_requested:
            # TODO - figure out logic here
            pass

        # Since we can use this function as an external callback for the
        # niDAQmx library, we need to guard against repeated calls to the
        # method after we have determined the current ABR is done.
        if not self.done:
            threshold = self.get_current_value('reject_threshold')
            self.current_valid_repetitions = \
                self.model.data.current_channel.get_n(threshold)
            if self.current_valid_repetitions > \
                    self.get_current_value('averages'):
                self.done = True
                if not self.stop_requested:
                    self.next_stimulus()
                else:
                    self.state = 'halted'
                    self.model.data.save()

    def set_frequency(self, value):
        self.current_graph.set_value('tone.frequency', value)

    def set_duration(self, value):
        self.current_graph.set_value('envelope.duration', value)

    def set_ramp_duration(self, value):
        self.current_graph.set_value('envelope.rise_time', value)

    def set_level(self, value):
        self.current_graph.set_value('tone.level', value)

    def set_repetition_rate(self, value):
        self.current_graph.set_value('sink.duration', 1.0/value)

    def set_window(self, value):
        self.iface_adc.epoch_duration = value
        self.model.data.epoch_size = int(value * self.adc_fs)

    def set_reject_threshold(self, value):
        self.model.plot.reject_threshold = value

    def calibrate_system(self, info=None):
        import calibration_chirp
        calibration_chirp.launch_gui(parent=info.ui.control)


class ABRExperiment(AbstractExperiment):

    paradigm = Instance(ABRParadigm, ())
    data = Instance(AbstractData, ())
    plot = Instance(Component)

    def _plot_default(self):
        index_mapper = LinearMapper(range=DataRange1D(low=0, high=10e-3))
        value_mapper = LinearMapper(range=DataRange1D(low=-3.0, high=3.0))
        plot = EpochChannelPlot(source=self.data.waveforms,
                                value_mapper=value_mapper,
                                index_mapper=index_mapper, bgcolor='white',
                                padding=[100, 50, 50, 75])
        axis = PlotAxis(orientation='left', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Signal (mV)')
        plot.overlays.append(axis)
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.overlays.append(axis)
        zoom = ZoomTool(plot, drag_button=None, axis="value")
        plot.overlays.append(zoom)
        return plot

    traits_view = View(
        HSplit(
            VGroup(
                VGroup(
                    Item('paradigm', style='custom', show_label=False,
                         width=200,
                         enabled_when='not handler.state=="running"'),
                    label='Paradigm',
                ),
                VGroup(
                    Item('handler.current_repetitions', style='readonly',
                         label='Repetitions'),
                    Item('handler.current_valid_repetitions', style='readonly',
                         label='Valid repetitions')
                ),
                show_border=True,
            ),
            VGroup(
                Item('plot', editor=ComponentEditor(width=250, height=250)),
                show_labels=False,
            ),
        ),
        resizable=True,
        height=500,
        width=800,
        toolbar=ToolBar(
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            Action(name='Pause', action='pause',
                   image=ImageResource('player_pause', icon_dir),
                   enabled_when='handler.state=="running"'),
            '-',
            Action(name='Apply', action='apply',
                   image=ImageResource('system_run', icon_dir),
                   enabled_when='handler.pending_changes'),
            Action(name='Revert', action='revert',
                   image=ImageResource('undo', icon_dir),
                   enabled_when='handler.pending_changes'),
        ),
        menubar=MenuBar(
            Menu(
                ActionGroup(
                    Action(name='Load settings', action='load_settings'),
                    Action(name='Save settings', action='save_settings'),
                ),
                name='&Settings',
            ),
            Menu(
                ActionGroup(
                    Action(name='Load system calibration',
                           action='load_system_calibration'),
                ),
                ActionGroup(
                    Action(name='Calibrate reference microphone',
                           action='calibrate_reference_microphone'),
                    Action(name='Calibrate system',
                           action='calibrate_system'),
                    Action(name='In-ear calibration',
                           action='calibrate_in_ear'),
                ),
                name='&Calibration'
            ),

        ),
        id='lbhb.ABRExperiment',
    )


if __name__ == '__main__':
    import PyDAQmx as ni
    ni.DAQmxResetDevice('Dev1')
    import logging
    #logging.basicConfig(level='DEBUG')
    with tables.open_file('test.hd5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        experiment = ABRExperiment(data=data, paradigm=ABRParadigm())
        experiment.configure_traits(handler=ABRController())
