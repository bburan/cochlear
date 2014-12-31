import logging
log = logging.getLogger(__name__)

from traits.api import Instance, Float, Int, push_exception_handler, Bool
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       OverlayPlotContainer, create_line_plot, ArrayPlotData,
                       Plot)

import numpy as np
from scipy import signal

from neurogen.block_definitions import Tone, Cos2Envelope, Click
from neurogen.calibration import PointCalibration

from cochlear import nidaqmx as ni
from cochlear import tone_calibration as tc

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, depends_on)
from experiment.coroutine import coroutine, blocked, counter

from experiment.plots.epoch_channel_plot import EpochChannelPlot

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 200e3
ADC_FS = 200e3

################################################################################
# Utility functions
################################################################################
@coroutine
def abr_reject(reject_threshold, fs, target):
    Wn = 0.2e3/fs, 10e3/fs
    b, a = signal.iirfilter(output='ba', N=1, Wn=Wn, btype='band',
                            ftype='butter')
    while True:
        data = (yield)
        d = signal.filtfilt(b, a, data, axis=-1)
        if np.all(np.max(np.abs(d), axis=-1) < reject_threshold):
            target.send(data)


################################################################################
# GUI class
################################################################################
class ABRParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    # Signal acquisition settings
    averages = Expression(512, dtype=np.int, **kw)
    window = Expression(8.5e-3, dtype=np.float, **kw)
    reject_threshold = Expression(2, dtype=np.float, **kw)
    exp_mic_gain = Expression(40, dtype=np.float, **kw)

    # Stimulus settings
    repetition_rate = Expression(20, dtype=np.float, **kw)
    frequency = Expression(8e3, dtype=np.float, **kw)
    duration = Expression(5e-3, dtype=np.float, **kw)
    ramp_duration = Expression(0.5e-3, dtype=np.float, **kw)
    level = Expression(
        'exact_order([20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90], c=1)',
        dtype=np.float, **kw)

    traits_view = View(
        VGroup(
            VGroup(
                'averages',
                'window',
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
            VGroup(
                'exp_mic_gain',
                label='Hardware settings',
                show_border=True,
            ),
        ),
    )


class ABRController(AbstractController):

    mic_cal = Instance('neurogen.calibration.Calibration')
    primary_sens = Instance('neurogen.calibration.PointCalibration')
    secondary_sens = Instance('neurogen.calibration.PointCalibration')
    iface_adc = Instance('cochlear.nidaqmx.TriggeredDAQmxSource')
    iface_dac = Instance('cochlear.nidaqmx.QueuedDAQmxPlayer')

    primary_spl = Float(label='Primary output @ 1Vrms, 0 dB gain (dB SPL)',
                        log=True, dtype=np.float32)

    current_valid_repetitions = Int(0)
    current_repetitions = Int(0)

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)
    done = Bool(False)

    extra_dtypes = [
        ('primary_sens', np.float32),
        ('primary_spl', np.float32),
    ]

    search_gains = [-20, -40, 0]

    @depends_on('exp_mic_gain')
    def set_frequency(self, frequency):
        log.debug('Calibrating primary speaker')
        self.primary_sens = tc.tone_calibration_search(
            frequency, self.mic_cal, self.search_gains, max_thd=0.1,
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT)
        self.primary_spl = self.primary_sens.get_spl(frequency, 1)

    def set_exp_mic_gain(self, exp_mic_gain):
        # Allow the calibration to automatically handle the gain.  Since this is
        # an input gain, it must be negative.
        self.mic_cal.set_fixed_gain(-exp_mic_gain)

    def stop_experiment(self, info=None):
        self.iface_dac.stop()
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.iface_atten.clear()
        self.model.data.save()

    def update_repetitions(self, value):
        self.current_repetitions = value

    def next_trial(self):
        try:
            self.refresh_context(evaluate=True)
        except StopIteration:
            self.stop()
            return

        frequency = self.get_current_value('frequency')
        level = self.get_current_value('level')
        duration = self.get_current_value('duration')
        ramp_duration = self.get_current_value('ramp_duration')

        token = Tone(name='tone', frequency=frequency, level=level) >> \
            Cos2Envelope(rise_time=ramp_duration, duration=duration)

        averages = self.get_current_value('averages')
        repetition_rate = self.get_current_value('repetition_rate')
        epoch_duration = self.get_current_value('window')
        reject_threshold = self.get_current_value('reject_threshold')

        # Pipeline to group acquisitions into pairs of two (i.e. alternate
        # polarity tone-pips), reject the pair if either exceeds artifact
        # reject, and accumulate the specified averages.  When the specified
        # number of averages are acquired, the program exits.
        pipeline = counter(self.update_repetitions,
                blocked(2, 0,
                abr_reject(reject_threshold, self.adc_fs,
                self)))

        # Set up the hardware
        iface_atten = ni.DAQmxAttenControl()
        iface_adc = ni.TriggeredDAQmxSource(
            epoch_duration=epoch_duration,
            input_line=ni.DAQmxDefaults.ERP_INPUT,
            pipeline=pipeline,
            fs=self.adc_fs,
            complete_callback=self.trial_complete)

        iface_dac = ni.QueuedDAQmxPlayer(
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
            duration=1.0/repetition_rate)

        channel = ni.DAQmxChannel(
            token=token,
            calibration=self.primary_sens,
            attenuator=iface_atten,
            attenuator_channel=ni.DAQmxDefaults.PRIMARY_ATTEN_CHANNEL)

        iface_dac.add_channel(channel, name='primary')
        iface_atten.setup()
        iface_dac.set_best_attenuations()
        iface_atten.clear()

        # Set up alternating polarity by shifting the phase np.pi.  Use the
        # Interleaved FIFO queue for this.
        iface_dac.queue_init('Interleaved FIFO')
        iface_dac.queue_append(np.inf, values={'primary.tone.phase': 0})
        iface_dac.queue_append(np.inf, values={'primary.tone.phase': np.pi})

        self.waveforms = []
        self.to_acquire = averages
        self.current_valid_repetitions = 0

        iface_adc.start()
        iface_dac.play_queue()

        self.iface_adc = iface_adc
        self.iface_dac = iface_dac
        self.iface_atten = iface_atten
        self.pipeline = pipeline

    def send(self, waveforms):
        self.waveforms.append(waveforms)
        self.current_valid_repetitions += 2
        self.current_repetitions = self.pipeline.n
        if self.current_valid_repetitions >= self.to_acquire:
            raise GeneratorExit

    def trial_complete(self):
        self.iface_dac.stop()
        self.iface_dac.clear()
        self.iface_adc.clear()
        waveforms = np.concatenate(self.waveforms, axis=0)[:self.to_acquire, 0]
        self.model.update_plots(self.adc_fs, waveforms)
        self.save_waveforms(waveforms)

        # If user has requested a pause, don't go to the next trial
        if not self.pause_requested:
            self.next_trial()
        else:
            self.state = 'paused'

    def save_waveforms(self, waveforms):
        primary_sens = self.primary_sens.get_sens(
            self.get_current_value('frequency'))
        self.log_trial(waveforms=waveforms, primary_sens=primary_sens,
                       primary_spl=self.primary_spl, fs=self.adc_fs)


class ABRExperiment(AbstractExperiment):

    paradigm = Instance(ABRParadigm, ())
    data = Instance(AbstractData, ())
    abr_stack = Instance(Component)
    neural_stack = Instance(Component)
    initialized = Bool(False)

    def _abr_stack_default(self):
        return VPlotContainer(padding=40, spacing=10)

    def _neural_stack_default(self):
        return VPlotContainer(padding=40, spacing=10)

    def update_plots(self, fs, waveforms):
        Wn = 0.2e3/fs, 10e3/fs
        b, a = signal.iirfilter(output='ba', N=1, Wn=Wn, btype='band',
                                ftype='butter')
        waveforms = signal.filtfilt(b, a, waveforms)

        time = np.arange(waveforms.shape[-1])/fs
        abr = np.mean(waveforms, axis=0)
        pos = np.mean(waveforms[::2], axis=0)
        neg = np.mean(waveforms[1::2], axis=0)
        neural = 0.5*(pos-neg)

        abr_plot = create_line_plot((time, abr), color='black')
        neural_plot = create_line_plot((time, neural), color='red')

        if not self.initialized:
            axis = PlotAxis(
                orientation='bottom', component=abr_plot,
                tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                title='Time (msec)')
            abr_plot.overlays.append(axis)
            axis = PlotAxis(
                orientation='bottom', component=neural_plot,
                tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                title='Time (msec)')
            neural_plot.overlays.append(axis)
            self.initialized = True

        axis = PlotAxis(orientation='left', component=abr_plot,
                        title='ABR (mv)')
        abr_plot.overlays.append(axis)
        axis = PlotAxis(orientation='right', component=neural_plot,
                        title='Neural (mv)')
        neural_plot.overlays.append(axis)

        self.abr_stack.add(abr_plot)
        self.abr_stack.request_redraw()
        self.neural_stack.add(neural_plot)
        self.neural_stack.request_redraw()

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
                    Item('handler.primary_spl', style='readonly',
                         format_str='%0.2f'),
                    Item('handler.current_repetitions', style='readonly',
                         label='Repetitions'),
                    Item('handler.current_valid_repetitions', style='readonly',
                         label='Valid repetitions')
                ),
                show_border=True,
            ),
            HGroup(
                Item('abr_stack', show_label=False,
                    editor=ComponentEditor(width=100, height=300)),
                Item('neural_stack', show_label=False,
                    editor=ComponentEditor(width=100, height=300)),
            ),
        ),
        resizable=True,
        height=500,
        width=300,
        toolbar=ToolBar(
            '-',  # hack to get below group to appear first
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            Action(name='Pause', action='request_pause',
                   image=ImageResource('player_pause', icon_dir),
                   enabled_when='handler.state=="running" and '
                                'not handler.pause_requested'),
            Action(name='Resume', action='resume',
                   image=ImageResource('player_fwd', icon_dir),
                   enabled_when='handler.state=="paused"'),
        ),
        id='lbhb.ABRExperiment',
    )


def launch_gui(mic_cal, filename, **kwargs):
    with tables.open_file(filename, 'w') as fh:
        data = ABRData(store_node=fh.root)
        paradigm = ABRParadigm()
        experiment = ABRExperiment(paradigm=paradigm, data=data)
        controller = ABRController(mic_cal=mic_cal)
        experiment.edit_traits(handler=controller, **kwargs)


class ABRData(AbstractData):

    waveform_node = Instance('tables.EArray')

    def log_trial(self, waveforms, fs, **kwargs):
        index = super(ABRData, self).log_trial(**kwargs)
        if self.waveform_node is None:
            shape = [0] + list(waveforms.shape)
            self.waveform_node = self.fh.create_earray(
                self.store_node, 'waveforms', tables.Float32Atom(), shape)
            self.waveform_node._v_attrs['fs'] = fs
        self.waveform_node.append(waveforms[np.newaxis])


if __name__ == '__main__':
    from cochlear import configure_logging
    from neurogen.calibration import InterpCalibration
    import PyDAQmx as pyni

    #configure_logging('temp.log')
    pyni.DAQmxResetDevice('Dev1')
    mic_file = 'c:/data/cochlear/calibration/141230 chirp calibration.mic'
    c = InterpCalibration.from_mic_file(mic_file)
    log.debug('====================== MAIN =======================')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        experiment = ABRExperiment(paradigm=ABRParadigm(), data=data)
        controller = ABRController(mic_cal=c)
        experiment.configure_traits(handler=controller)
