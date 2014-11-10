from traits.api import (Instance, Float, push_exception_handler, Bool, Property,
                        cached_property, Int)
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include)

from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       create_line_plot,  LogMapper)

import numpy as np

from neurogen.block_definitions import Tone, Cos2Envelope
from neurogen.util import db
from cochlear.nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxPlayer,
                              DAQmxChannel, DAQmxAttenControl)

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment)
from experiment.channel import FileEpochChannel

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 200e3
ADC_FS = 200e3


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
                                   dtype=np.double)
        self.current_channel = channel
        return channel


class DPOAEParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    dpoae_frequency = Expression('2*f2_frequency-f1_frequency', **kw)
    f1_frequency = Expression('f2_frequency/1.2', **kw)
    f2_frequency = Expression('8e3', **kw)
    f1_level = Expression('f2_level+10', **kw)
    f2_level = Expression(70, **kw)

    probe_duration = Expression('50e-3', **kw)
    response_window = Expression('probe_duration', **kw)
    ramp_duration = Expression(0.5e-3, **kw)
    iti = Expression(0.01, label='Intertrial interval (sec)', **kw)
    exp_mic_gain = Float(40, label='Exp. mic. gain (dB)', **kw)

    # Signal acquisition settings.  Increasing time_averages increases SNR by
    # sqrt(N).  Increasing spectral averages reduces variance of result.
    time_averages = Float(8, **kw)
    spectrum_averages = Float(4, **kw)

    n_stimuli = Property(depends_on='time_averages, spectrum_averages', **kw)

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
                'exp_mic_gain',
                label='Acquisition settings',
                show_border=True,
            ),
            VGroup(
                'f1_frequency',
                'f2_frequency',
                'dpoae_frequency',
                'f1_level',
                'f2_level',
                'ramp_duration',
                label='Stimulus settings',
                show_border=True,
            ),
        ),
    )


class DPOAEController(DAQmxDefaults, AbstractController):

    inear_cal_0 = Instance('neurogen.calibration.SimpleCalibration')
    inear_cal_1 = Instance('neurogen.calibration.SimpleCalibration')
    iface_adc = Instance('cochlear.nidaqmx.TriggeredDAQmxSource')
    iface_dac = Instance('cochlear.nidaqmx.DAQmxPlayer')

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)

    done = Bool(False)
    stop_requested = Bool(False)

    current_repetitions = Int(0)

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
        duration = self.get_current_value('probe_duration')
        response_window = self.get_current_value('response_window')
        ramp_duration = self.get_current_value('ramp_duration')
        iti = self.get_current_value('iti')

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

        self.iface_dac = DAQmxPlayer(fs=self.dac_fs,
                                     output_line=self.DUAL_SPEAKER_OUTPUT,
                                     trigger_line=self.SPEAKER_TRIGGER,
                                     run_line=self.SPEAKER_RUN,
                                     duration=duration+iti)

        c0 = Tone(frequency=f1_frequency, level=f1_level, name='f1') >> \
            Cos2Envelope(duration=duration, rise_time=ramp_duration) >> \
            DAQmxChannel(calibration=self.inear_cal_0,
                         attenuator=self.iface_atten,
                         attenuator_channel='left')
        c1 = Tone(frequency=f2_frequency, level=f2_level, name='f2') >> \
            Cos2Envelope(duration=duration, rise_time=ramp_duration) >> \
            DAQmxChannel(calibration=self.inear_cal_1,
                         attenuator=self.iface_atten,
                         attenuator_channel='right')

        # Ordering is important.  First channel is sent to ao0, second channel
        # to ao1.  The left attenuator channel controls ao0, right attenuator
        # channel controls ao1.
        self.iface_dac.add_channel(c0)
        self.iface_dac.add_channel(c1)

        self.model.data.create_waveform_channel(self.current_trial, self.adc_fs,
                                                response_window)

        self.iface_atten.setup()
        print 'attens (L then R, f1 then f2)', \
            self.iface_dac.set_best_attenuations()
        self.iface_atten.clear()

        self.iface_dac.queue_init('FIFO')
        self.iface_dac.queue_append(n_stimuli)

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
        self.current_repetitions += len(waveform)

        #time_averages = self.get_current_value('time_averages')
        #if (self.current_repetitions % time_averages) == 0:

        if self.current_repetitions >= self.get_current_value('n_stimuli'):
            self.done = True
            if not self.stop_requested:
                self.iface_adc.clear()
                self.iface_dac.clear()
                self.iface_atten.clear()
                self.model.update_plots()
                self.next_stimulus()
            else:
                self.stop()


class DPOAEExperiment(AbstractExperiment):

    paradigm = Instance(DPOAEParadigm, ())
    data = Instance(AbstractData, ())
    exp_mic_sens = Instance('numpy.ndarray')

    container = Instance(Component)

    def update_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        channel = self.data.current_channel
        x = np.arange(channel.epoch_size)/channel.fs
        y = channel.get_average()*1e3
        plot = create_line_plot((x, y), color='black')
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (mV)')
        plot.underlays.append(axis)
        container.insert(0, plot)

        index_range = DataRange1D(low_setting=500, high_setting=50e3)
        index_mapper = LogMapper(range=index_range)
        frequency = channel.get_fftfreq()
        averages = self.paradigm.time_averages

        mic_rms = channel.get_average_psd(waveform_averages=averages, rms=True)
        plot = create_line_plot((frequency[1:], db(mic_rms[1:], 1e-3)),
                                color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='Frequency (Hz)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (dB re 1mV)')
        plot.underlays.append(axis)
        container.insert(0, plot)

        #inear_db = self.mic_cal.get_spl(frequency, mic_vrms)
        #inear_db_spl = inear_db-self.paradigm.exp_mic_gain
        #plot = create_line_plot((frequency[1:], inear_db_spl[1:]),
        #                        color='black')
        #plot.index_mapper = index_mapper
        #axis = PlotAxis(orientation='bottom', component=plot,
        #                title='Frequency (Hz)')
        #plot.underlays.append(axis)
        #axis = PlotAxis(orientation='left', component=plot, title='Inear SPL')
        #plot.underlays.append(axis)
        #container.insert(0, plot)
        self.container = container

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
                ),
                show_border=True,
            ),
            Item('container', show_label=False,
                 editor=ComponentEditor(width=300, height=600)),
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


def launch_gui(inear_cal_0, inear_cal_1, exp_mic_sens, **kwargs):
    with tables.open_file('test.hd5', 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        experiment = DPOAEExperiment(data=data, paradigm=DPOAEParadigm(),
                                     exp_mic_sens=exp_mic_sens)
        controller = DPOAEController(inear_cal_0=inear_cal_0,
                                     inear_cal_1=inear_cal_1)
        experiment.edit_traits(handler=controller, **kwargs)
