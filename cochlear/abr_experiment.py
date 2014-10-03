from traits.api import Instance, Float, Int, push_exception_handler, Bool
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       create_line_plot)

import numpy as np

from neurogen.block_definitions import Tone, Cos2Envelope
from neurogen.calibration import Attenuation
from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)


from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, util)
from experiment.channel import FileEpochChannel

from experiment.plots.epoch_channel_plot import EpochChannelPlot

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 200e3
ADC_FS = 50e3


class ABRData(AbstractData):

    current_channel = Instance('experiment.channel.Channel')
    waveform_node = Instance('tables.Group')

    def _waveform_node_default(self):
        return self.fh.create_group(self.store_node, 'waveforms')

    def create_waveform_channel(self, trial, fs, epoch_duration):
        # We use a separate "channel" for each trial set.
        channel = FileEpochChannel(node=self.waveform_node,
                                   name='stack_{}'.format(trial),
                                   epoch_duration=epoch_duration, fs=fs,
                                   dtype=np.double, use_checksum=True)
        self.current_channel = channel
        return channel


class ABRParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    # Signal acquisition settings
    averages = Expression(10, **kw)
    window = Expression(10e-3, **kw)
    quick_threshold = Expression(550e-9, **kw)
    reject_threshold = Expression(15, **kw)

    # Stimulus settings
    repetition_rate = Expression(40, **kw)
    frequency = Expression(1e3, **kw)
    duration = Expression(4e-3, **kw)
    ramp_duration = Expression(0.5e-3, **kw)
    level = Expression('ascending(np.arange(5, 80, 5), cycles=1)', **kw)

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
        pass

    def start_experiment(self, info=None):
        self.next_stimulus()

    def stop_experiment(self, info=None):
        self.stop_requested = True

    def stop(self, info=None):
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.iface_atten.clear()
        self.state = 'halted'
        self.model.data.save()

    def next_stimulus(self):
        try:
            self.refresh_context(evaluate=True)
        except StopIteration:
            # We are done with the experiment
            self.stop()
            return

        epoch_duration = self.get_current_value('window')
        frequency = self.get_current_value('frequency')
        duration = self.get_current_value('duration')
        ramp_duration = self.get_current_value('ramp_duration')
        level = self.get_current_value('level')
        repetition_rate = self.get_current_value('repetition_rate')

        self.iface_atten = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                             cs_line=self.VOLUME_CS,
                                             data_line=self.VOLUME_SDI,
                                             mute_line=self.VOLUME_MUTE,
                                             zc_line=self.VOLUME_ZC,
                                             hw_clock=self.DIO_CLOCK)
        self.iface_dac = DAQmxSink(name='sink', fs=self.dac_fs,
                                   calibration=Attenuation(),
                                   output_line=self.SPEAKER_OUTPUT,
                                   trigger_line=self.SPEAKER_TRIGGER,
                                   run_line=self.SPEAKER_RUN,
                                   attenuator=self.iface_atten,
                                   duration=1.0/repetition_rate)
        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.ERP_INPUT,
                                              counter_line=self.ERP_COUNTER,
                                              trigger_line=self.ERP_TRIGGER,
                                              callback=self.poll)

        tone = Tone(frequency=frequency, level=level, name='tone')
        envelope = Cos2Envelope(duration=duration, rise_time=ramp_duration)
        self.current_graph = tone >> envelope >> self.iface_dac
        self.iface_adc.setup()
        self.model.abr_current.source = \
            self.model.data.create_waveform_channel(self.current_trial,
                                                    self.adc_fs, epoch_duration)

        # Set up alternating polarity by shifting the phase np.pi.  Use the
        # Interleaved FIFO queue for this.
        self.current_graph.queue_init('Interleaved FIFO')
        self.current_graph.set_value('tone.phase', 0)
        self.current_graph.queue_append(np.inf)
        self.current_graph.set_value('tone.phase', np.pi)
        self.current_graph.queue_append(np.inf)

        self.done = False
        self.current_trial += 1
        self.current_repetitions = 0
        self.current_valid_repetitions = 0
        self.iface_adc.start()
        self.current_graph.play_queue()

    def poll(self):
        # Since we can use this function as an external callback for the niDAQmx
        # library, we need to guard against repeated calls to the method after
        # we have determined the current ABR is done.
        if self.done:
            return

        # Read in new data
        waveform = self.iface_adc.read_analog(timeout=-1)
        self.model.data.current_channel.send(waveform)
        self.current_repetitions += 1

        threshold = self.get_current_value('reject_threshold')
        self.current_valid_repetitions = \
            self.model.data.current_channel.get_n(threshold)
        if self.current_valid_repetitions > \
                self.get_current_value('averages'):
            self.done = True
            if not self.stop_requested:
                self.iface_adc.clear()
                self.iface_dac.clear()
                self.iface_atten.clear()
                self.model.add_plot_to_stack()
                self.next_stimulus()
            else:
                self.stop()

    def set_reject_threshold(self, value):
        self.model.abr_current.reject_threshold = value

    def calibrate_system(self, info=None):
        import calibration_chirp
        calibration_chirp.launch_gui(parent=info.ui.control)

    def load_microphone_calibration(self):
        print util.get_save_file('*_miccal.hdf5')


class ABRExperiment(AbstractExperiment):

    paradigm = Instance(ABRParadigm, ())
    data = Instance(AbstractData, ())
    abr_current = Instance(Component)
    abr_stack = Instance(Component)

    def _abr_current_default(self):
        index_mapper = LinearMapper(range=DataRange1D(low=0, high=10e-3))
        value_mapper = LinearMapper(range=DataRange1D(low=-3.0, high=3.0))
        plot = EpochChannelPlot(value_mapper=value_mapper,
                                index_mapper=index_mapper,
                                bgcolor='white',
                                update_rate=2,
                                padding=[100, 50, 50, 75])
        axis = PlotAxis(orientation='left', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Signal (mV)')
        plot.overlays.append(axis)
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.overlays.append(axis)
        return plot

    def _abr_stack_default(self):
        return VPlotContainer(padding=50, spacing=25)

    def add_plot_to_stack(self):
        channel = self.data.current_channel
        x = np.arange(channel.epoch_size)/channel.fs
        y = channel.get_average()
        plot = create_line_plot((x, y))
        self.abr_stack.add(plot)
        self.abr_stack.request_redraw()

    traits_view = View(
        HSplit(
            VGroup(
                Tabbed(
                    Item('paradigm', style='custom', show_label=False,
                         width=200,
                         enabled_when='not handler.state=="running"'),
                    Include('context_group'),
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
            HGroup(
                Item('abr_current',
                     editor=ComponentEditor(width=100, height=100)),
                Item('abr_stack',
                     editor=ComponentEditor(width=100, height=300)),
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
                    Action(name='Load microphone calibration',
                           action='load_mic_cal'),
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
    with tables.open_file('test.hd5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        experiment = ABRExperiment(data=data, paradigm=ABRParadigm())
        experiment.configure_traits(handler=ABRController())
