from traits.api import Instance, Float, push_exception_handler, Bool
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include,
                          Property, cached_property)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       create_line_plot)

import numpy as np

from neurogen.block_definitions import Tone, Cos2Envelope
from cochlear.nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource,
                              DAQmxMultiSink, DAQmxAttenControl)


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


class DPOAEData(AbstractData):

    current_channel = Instance('experiment.channel.Channel')
    waveform_node = Instance('tables.Group')

    def _waveform_node_default(self):
        return self.fh.create_group(self.store_node, 'waveforms')

    def create_waveform_channel(self, trial, fs, epoch_duration):
        # We use a separate "channel" for each trial set.
        channel = FileEpochChannel(node=self.waveform_node,
                                   name='stack_{}'.format(trial),
                                   epoch_duration=epoch_duration, fs=fs,
                                   dtype=np.double, use_checksum=True,
                                   use_shuffle=True, compression_level=9,
                                   compression_type='blosc')
        self.current_channel = channel
        return channel


class DPOAEParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    f1_frequency = Expression('f2_frequency/1.2', **kw)
    f2_frequency = Expression('8e3', **kw)
    f1_level = Expression('f2_level+10', **kw)
    f2_level = Expression(70, **kw)

    probe_duration = Expression('50e-3', **kw)
    response_window = Expression('probe_duration', **kw)
    ramp_duration = Expression(0.5e-3, **kw)

    # Signal acquisition settings.  Increasing time_averages increases SNR by
    # sqrt(N).  Increasing spectral averages reduces variance of result.
    time_averages = Expression(8, **kw)
    spectrum_averages = Expression(4, **kw)

    n_stimuli = Property(depends_on='*_averages')

    @cached_property
    def _get_n_stimuli(self):
        return self.time_averages*self.spectrum_averages

    traits_view = View(
        VGroup(
            VGroup(
                'time_averages',
                'spectrum_averages',
                Item('n_stimuli', style='readonly'),
                'probe_duration',
                'response_window',
                'ramp_duration',
                label='Acquisition settings',
                show_border=True,
            ),
            VGroup(
                'f1_frequency',
                'f2_frequency',
                'f1_level',
                'f2_level',
                label='Stimulus settings',
                show_border=True,
            ),
        ),
    )


class DPOAEController(DAQmxDefaults, AbstractController):

    inear_cal = Instance('neurogen.calibration.SimpleCalibration')
    iface_adc = Instance('cochlear.nidaqmx.TriggeredDAQmxSource')
    iface_dac = Instance('cochlear.nidaqmx.DAQmxSink')

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)

    done = Bool(False)
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
        if self.iface_dac is not None:
            self.iface_dac.clear()
            self.iface_adc.clear()
            self.iface_atten.clear()

        try:
            self.refresh_context(evaluate=True)
        except StopIteration:
            # We are done with the experiment
            self.stop()
            return

        f1_frequency = self.get_current_value('f1_frequency')
        f2_frequency = self.get_current_value('f2_frequency')
        f1_level = self.get_current_value('f1_level')
        f2_level = self.get_current_value('f2_level')
        n_stimuli = self.get_current_value('n_stimuli')
        duration = self.get_current_value('duration')
        response_window = self.get_current_value('response_window')
        ramp_duration = self.get_current_value('ramp_duration')

        self.iface_atten = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                             cs_line=self.VOLUME_CS,
                                             data_line=self.VOLUME_SDI,
                                             mute_line=self.VOLUME_MUTE,
                                             zc_line=self.VOLUME_ZC,
                                             hw_clock=self.DIO_CLOCK)

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=response_window,
                                              input_line=self.MIC_INPUT,
                                              counter_line=self.AI_COUNTER,
                                              trigger_line=self.AI_TRIGGER,
                                              callback=self.poll)

        self.iface_dac = DAQmxMultiSink(name='sink',
                                        fs=self.dac_fs,
                                        output_line=self.DUAL_SPEAKER_OUTPUT,
                                        trigger_line=self.SPEAKER_TRIGGER,
                                        run_line=self.SPEAKER_RUN,
                                        attenuator=self.iface_atten,
                                        duration=duration)

        f1 = Tone(frequency=f1_frequency, level=f1_level, name='f1')
        e1 = Cos2Envelope(duration=duration, rise_time=ramp_duration, token=f1)
        f2 = Tone(frequency=f2_frequency, level=f2_level, name='f2')
        e2 = Cos2Envelope(duration=duration, rise_time=ramp_duration, token=f2)
        s1 = Sink(token=e1, calibration=self.inear_cal_1)
        s2 = Sink(token=e2, calibration=self.inear_cal_2)
        self.iface_dac.append_sink(s1, 'right')
        self.iface_dac.append_sink(s2, 'left')

        self.model.data.create_waveform_channel(self.current_trial, self.adc_fs,
                                                response_window)

        a1, a2 = self.iface_dac.get_best_attens()
        self.iface_atten.setup()
        self.iface_atten.set_atten(a1, a2)
        self.iface_atten.clear()

        # Set up alternating polarity by shifting the phase np.pi.  Use the
        # Interleaved FIFO queue for this.
        self.current_graph.queue_init('FIFO')
        self.current_graph.queue_append(n_stimuli)

        self.done = False
        self.current_trial += 1
        self.current_repetitions = 0
        self.iface_adc.setup()
        self.iface_adc.start()
        self.iface_dac.play_queue()

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

        if self.current_repetitions >= self.get_current_value('n_stimuli'):
            self.done = True
            if not self.stop_requested:
                self.iface_adc.clear()
                self.iface_dac.clear()
                self.iface_atten.clear()
                self.model.add_plot_to_stack()
                self.next_stimulus()
            else:
                self.stop()


class DPOAEExperiment(AbstractExperiment):

    paradigm = Instance(DPOAEParadigm, ())
    data = Instance(AbstractData, ())

    def add_plot_to_stack(self):
        channel = self.data.current_channel
        x = np.arange(channel.epoch_size)/channel.fs
        y = channel.get_average()
        plot = create_line_plot((x, y), color='black')
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.overlays.append(axis)
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
            '-',  # hack to get below group to appear first
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.inear_cal is not None'),
            Action(name='Stop', action='stop',
                   image=ImageResource('stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            Action(name='Pause', action='pause',
                   image=ImageResource('player_pause', icon_dir),
                   enabled_when='handler.state=="running"'),
        ),
        menubar=MenuBar(
            Menu(
                ActionGroup(
                    Action(name='Load settings', action='load_settings'),
                    Action(name='Save settings', action='save_settings'),
                ),
                name='&Settings',
            ),
        ),
        id='lbhb.ABRExperiment',
    )


def launch_gui(inear_cal, **kwargs):
    with tables.open_file('test.hd5', 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        experiment = DPOAEExperiment(data=data, paradigm=ABRParadigm())
        #controller = DPOAEController(inear_cal=inear_cal)
        #experiment.edit_traits(handler=controller, **kwargs)
        experiment.edit_traits(**kwargs)
