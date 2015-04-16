import logging
log = logging.getLogger(__name__)

from traits.api import (Instance, Float, Int, push_exception_handler, Bool,
                        HasTraits, Str, List, Enum)
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include,
                          ListEditor)
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
    averages = Expression(1024, dtype=np.int, **kw)
    window = Expression(8.5e-3, dtype=np.float, **kw)
    reject_threshold = Expression(0.2, dtype=np.float, **kw)
    exp_mic_gain = Expression(40, dtype=np.float, **kw)

    # Stimulus settings
    repetition_rate = Expression(20, dtype=np.float, **kw)
    repetition_jitter = Expression(0, dtype=np.float, **kw)

    frequencies = [4000, 5660, 8000, 11310, 16000, 22630]
    frequency = Expression('u(exact_order({}, c=1), level)'.format(frequencies),
                           dtype=np.float, **kw)
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
                Item('repetition_jitter', label='Jitter in rep. rate (frac)'),
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

    kw = dict(log=True, dtype=np.float32)
    primary_spl = Float(label='Primary @ 1Vrms, 0dB att (dB SPL)', **kw)
    primary_attenuation = Float(label='Primary attenuation (dB)', **kw)
    primary_calibration_gain = Float(label='Primary cal. gain (dB)', **kw)

    current_valid_repetitions = Int(0)
    current_repetitions = Int(0)

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)
    done = Bool(False)

    frequency_changed = Bool(False)

    extra_dtypes = [
        ('primary_sens', np.float32),
        ('primary_spl', np.float32),
        ('primary_attenuation', np.float32),
        ('primary_calibration_gain', np.float32),
        ('total_repetitions', np.float32),
    ]

    @depends_on('exp_mic_gain')
    def set_frequency(self, frequency):
        log.debug('Calibrating primary speaker')
        self.primary_sens = tc.tone_calibration(
            frequency, self.mic_cal, gain=-40, max_thd=None,
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT)
        self.primary_spl = self.primary_sens.get_spl(frequency, 1)
        self.frequency_changed = True

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
        repetition_jitter = self.get_current_value('repetition_jitter')

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
        iface_atten.setup()

        iface_adc = ni.TriggeredDAQmxSource(
            epoch_duration=epoch_duration,
            input_line=ni.DAQmxDefaults.ERP_INPUT,
            pipeline=pipeline,
            fs=self.adc_fs,
            expected_range=10,  # 10,000 gain
            complete_callback=self.trial_complete,
            record_mode=ni.DAQmxSource.RSE,
        )

        iface_dac = ni.QueuedDAQmxPlayer(
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
            duration=1.0/repetition_rate)

        channel = ni.DAQmxChannel(
            token=token,
            calibration=self.primary_sens,
            attenuator=iface_atten,
            attenuator_channel=ni.DAQmxDefaults.PRIMARY_ATTEN_CHANNEL)

        iface_dac.add_channel(channel, name='primary')
        self.primary_attenuation = iface_dac.set_best_attenuations()[0]

        # Set up alternating polarity by shifting the phase np.pi.  Use the
        # Interleaved FIFO queue for this.
        delay_func = lambda: np.random.uniform(low=0, high=repetition_jitter)
        iface_dac.queue_init('Interleaved FIFO')
        iface_dac.queue_append(np.inf, values={'primary.tone.phase': 0},
                               delays=delay_func)
        iface_dac.queue_append(np.inf, values={'primary.tone.phase': np.pi},
                               delays=delay_func)

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
        if self.frequency_changed:
            self.model.clear_abr_data(str(self.get_current_value('frequency')))
            self.frequency_changed = False
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
        self.log_trial(waveforms=waveforms,
                       primary_sens=primary_sens,
                       primary_spl=self.primary_spl,
                       fs=self.adc_fs,
                       primary_attenuation=self.primary_attenuation,
                       primary_calibration_gain=self.primary_calibration_gain,
                       total_repetitions=self.current_repetitions,
                       )

    def primary_calibration_gain_callback(self, value):
        self.primary_calibration_gain = value


class _ABRPlot(HasTraits):

    parameter = Str
    stack = Instance(Component)

    def _stack_default(self):
        return VPlotContainer(padding=40, spacing=10)

    traits_view = View(
        Item('stack', show_label=False,
             editor=ComponentEditor(width=300, height=300))
    )


class ABRExperiment(AbstractExperiment):

    paradigm = Instance(ABRParadigm, ())
    data = Instance(AbstractData, ())
    abr_waveforms = List(Instance(_ABRPlot))
    initialized = Bool(False)

    def clear_abr_data(self, parameter):
        self.initialized = False
        self.abr_waveforms.append(_ABRPlot(parameter=parameter))

    def update_plots(self, fs, waveforms):
        b, a = signal.iirfilter(output='ba', N=1, Wn=(0.2e3/fs, 10e3/fs),
                                btype='band', ftype='butter')
        waveforms = signal.filtfilt(b, a, waveforms)
        time = np.arange(waveforms.shape[-1])/fs
        abr = np.mean(waveforms, axis=0)
        plot = create_line_plot((time, abr), color='black')

        if not self.initialized:
            axis = PlotAxis(
                orientation='bottom', component=plot,
                tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                title='Time (msec)')
            plot.overlays.append(axis)
            self.initialized = True
        axis = PlotAxis(orientation='left', component=plot,
                        title='ABR (mv)')
        plot.overlays.append(axis)
        self.abr_waveforms[-1].stack.add(plot)
        self.abr_waveforms[-1].stack.request_redraw()

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
                    Item('handler.primary_calibration_gain',
                            style='readonly', format_str='%d'),
                    Item('handler.primary_spl', style='readonly',
                         format_str='%0.2f'),
                    Item('handler.primary_attenuation', style='readonly',
                         format_str='%0.2f'),
                    Item('handler.current_repetitions', style='readonly',
                         label='Repetitions'),
                    Item('handler.current_valid_repetitions', style='readonly',
                         label='Valid repetitions'),
                    label='Diagnostics',
                    show_border=True,
                ),
                show_border=True,
            ),
            HGroup(
                Item('abr_waveforms', show_label=False, style='custom',
                     editor=ListEditor(use_notebook=True, deletable=False,
                                       page_name='.parameter'))
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


def launch_gui(mic_cal, filename, paradigm_dict=None, **kwargs):
    if filename is None:
        filename = 'dummy'
        tbkw = {'driver': 'H5FD_CORE', 'driver_core_backing_store': 0}
    else:
        tbkw = {}
    with tables.open_file(filename, 'w', **tbkw) as fh:
        data = ABRData(store_node=fh.root)
        if paradigm_dict is None:
            paradigm_dict = {}
        paradigm = ABRParadigm(**paradigm_dict)
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

    pyni.DAQmxResetDevice('Dev1')
    mic_file = 'c:/data/cochlear/calibration/150407 - calibration with 377C10.mic'
    c = InterpCalibration.from_mic_file(mic_file)
    log.debug('====================== MAIN =======================')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        experiment = ABRExperiment(paradigm=ABRParadigm(), data=data)
        controller = ABRController(mic_cal=c)
        experiment.configure_traits(handler=controller)
