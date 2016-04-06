import logging
log = logging.getLogger(__name__)
import os.path

import time

from traits.api import (Instance, Float, Int, push_exception_handler, Bool,
                        HasTraits, Str, List, Property, on_trait_change)
from traitsui.api import (View, Item, ToolBar, Action, VGroup, HSplit, Tabbed,
                          Include, ListEditor)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       OverlayPlotContainer, create_line_plot)

import numpy as np
from scipy import signal

from cochlear import nidaqmx as ni
from cochlear import calibration

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, depends_on)
from experiment.channel import FileChannel

from experiment.plots.extremes_channel_plot import ExtremesChannelPlot
from experiment.plots.channel_data_range import ChannelDataRange
from experiment.plots.channel_range_tool import ChannelRangeTool
from experiment.plots.helpers import add_time_axis

from cochlear.acquire import ABRAcquisition

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 100e3
ADC_FS = 100e3


################################################################################
# GUI class
################################################################################
class ABRParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    # Signal acquisition settings
    averages = Expression(512, dtype=np.int, **kw)
    window = Expression(8.5e-3, dtype=np.float, **kw)
    reject_threshold = Expression(0.2, dtype=np.float, **kw)
    exp_mic_gain = Expression(40, dtype=np.float, **kw)

    # Stimulus settings
    repetition_rate = Expression(20, dtype=np.float, **kw)
    repetition_jitter = Expression(0, dtype=np.float, **kw)

    frequencies = [1420, 4000, 8000, 700, 2840]
    frequency = Expression('u(exact_order({}, c=1), level)'.format(frequencies),
                           dtype=np.float, **kw)
    duration = Expression(5e-3, dtype=np.float, **kw)
    ramp_duration = Expression(0.5e-3, dtype=np.float, **kw)
    level = Expression(
        'exact_order([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], c=1)',
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
    iface_adc = Instance('cochlear.nidaqmx.DAQmxInput')
    iface_dac = Instance('cochlear.nidaqmx.QueuedDAQmxPlayer')

    kw = dict(log=True, dtype=np.float32)
    primary_spl = Float(label='Primary @ 1Vrms, 0dB att (dB SPL)', **kw)
    primary_attenuation = Float(label='Primary attenuation (dB)', **kw)
    start_time = Float(label='Start time', **kw)
    end_time = Float(label='End time', **kw)

    current_valid_repetitions = Int(0, log=True, dtype=np.int32)
    current_repetitions = Int(0, log=True, dtype=np.int32)
    current_rejects = Int(0, log=True, dtype=np.int32)
    current_calibration_gain = Property(log=True, dtype=np.float32)

    adc_fs = Float(ADC_FS, **kw)
    dac_fs = Float(DAC_FS, **kw)
    done = Bool(False)

    frequency_changed = Bool(False)

    current_trial = Int(-1)

    token_sequence = Instance('collections.deque')

    raw_channel = Instance('experiment.channel.Channel')

    extra_dtypes = [
        ('primary_sens', np.float32),
    ]

    def _get_current_calibration_gain(self):
        frequency = self.get_current_value('frequency')
        return -20 if frequency <= 700 else -40

    @depends_on('exp_mic_gain')
    def set_frequency(self, frequency):
        log.debug('Calibrating primary speaker')
        self.primary_sens = calibration.tone_calibration(
            frequency, self.mic_cal, gain=self.current_calibration_gain,
            max_thd=None, output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT)
        self.primary_spl = self.primary_sens.get_spl(frequency, 1)
        self.frequency_changed = True

    def set_exp_mic_gain(self, exp_mic_gain):
        # Allow the calibration to automatically handle the gain.  Since this is
        # an input gain, it must be negative.
        self.mic_cal.set_fixed_gain(-exp_mic_gain)

    def stop_experiment(self, info=None):
        self.model.data.save()

    def update_repetitions(self, value):
        self.current_repetitions = value

    def next_trial(self):
        try:
            self.refresh_context(evaluate=True)
        except StopIteration:
            self.stop()
            return

        self.current_trial += 1
        self.model.data.raw_channel = \
            FileChannel(node=self.model.data.store_node, fs=self.adc_fs,
                        name='raw_data_{}'.format(self.current_trial))

        frequencies = [self.get_current_value('frequency')]
        levels = [self.get_current_value('level')]
        pip_averages = np.ceil(self.get_current_value('averages')/2.0)

        self.acq = ABRAcquisition(
            frequencies=frequencies,
            levels=levels,
            calibration=self.primary_sens,
            duration=self.get_current_value('duration'),
            ramp_duration=self.get_current_value('ramp_duration'),
            pip_averages=pip_averages,
            window=self.get_current_value('window'),
            reject_threshold=self.get_current_value('reject_threshold'),
            repetition_rate=self.get_current_value('repetition_rate'),
            adc_fs=self.adc_fs,
            dac_fs=self.dac_fs,
            samples_acquired_callback=self.samples_acquired,
            valid_epoch_callback=self.valid_epoch,
            invalid_epoch_callback=self.invalid_epoch,
            done_callback=self.acquisition_complete,
        )

        self.current_valid_repetitions = 0
        self.current_rejects = 0
        self.start_time = time.time()
        self.acq.start()

    def valid_epoch(self, frequency, level, polarity, presentation, epoch):
        self.current_valid_repetitions += 1

    def invalid_epoch(self, level, frequency, polarity, epoch):
        self.current_rejects += 1

    def samples_acquired(self, samples):
        self.model.data.raw_channel.send(samples.ravel())

    def acquisition_complete(self, state):
        if state == 'aborted':
            self.stop()
        elif state == 'complete':
            log.debug('Acquisition complete')
            frequency = self.get_current_value('frequency')
            level = self.get_current_value('level')
            if self.frequency_changed:
                log.debug('Frequency changed. Updating plots.')
                self.model.clear_abr_data(str(frequency))
                self.frequency_changed = False
            waveforms = self.acq.get_waveforms(frequency, level)[:, 0]
            self.model.update_plots(self.adc_fs, waveforms)
            self.save_waveforms(waveforms)

            # If user has requested a pause, don't go to the next trial
            if not self.pause_requested:
                log.debug('Starting next trial')
                self.next_trial()
            else:
                log.debug('Pausing')
                self.state = 'paused'

    def save_waveforms(self, waveforms):
        primary_sens = self.primary_sens.get_sens(
            self.get_current_value('frequency'))
        self.log_trial(waveforms=waveforms,
                       primary_sens=primary_sens,
                       primary_spl=self.primary_spl,
                       fs=self.adc_fs,
                       primary_attenuation=self.primary_attenuation,
                       primary_calibration_gain=self.current_calibration_gain,
                       total_repetitions=self.current_repetitions,
                       start_time=self.start_time,
                       end_time=self.end_time,
                       )

    def request_stop(self, info=None):
        self.acq.request_stop()
        self.acq.join()


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

    running_plot = Instance('enable.api.Component')

    @on_trait_change('data.raw_channel')
    def _update_raw_channel_plot(self, new):
        if new is None:
            return
        container = OverlayPlotContainer(padding=[75, 20, 20, 50])
        data_range = ChannelDataRange(sources=[new], span=2, trig_delay=0)
        index_mapper = LinearMapper(range=data_range)
        value_range = DataRange1D(low_setting=-1, high_setting=1)
        value_mapper = LinearMapper(range=value_range)
        plot = ExtremesChannelPlot(source=new, index_mapper=index_mapper,
                                   value_mapper=value_mapper)
        tool = ChannelRangeTool(plot)
        plot.tools.append(tool)
        axis = PlotAxis(component=plot, orientation='left', title='Volts (V)')
        plot.underlays.append(axis)
        add_time_axis(plot, 'bottom', fraction=True)
        container.add(plot)
        self.running_plot = container

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
                ),
                VGroup(
                    Item('handler.primary_spl', style='readonly',
                         format_str='%0.2f'),
                    Item('handler.primary_attenuation', style='readonly',
                         format_str='%0.2f'),
                    Item('handler.current_repetitions', style='readonly',
                         label='Repetitions'),
                    Item('handler.current_valid_repetitions', style='readonly',
                         label='Valid repetitions'),
                    Item('handler.current_rejects', style='readonly',
                         label='Rejects'),
                    label='Diagnostics',
                    show_border=True,
                ),
                show_border=True,
            ),
            VGroup(
                Item('object.running_plot',
                     editor=ComponentEditor(width=500, height=100), height=100),
                Item('abr_waveforms', style='custom',
                     editor=ListEditor(use_notebook=True, deletable=False,
                                       page_name='.parameter'), height=800),
                show_labels=False,
                id='foo.bar',
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
            Action(name='Stop', action='request_stop',
                   image=ImageResource('stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            '-',
            Action(name='Pause', action='request_pause',
                   image=ImageResource('player_pause', icon_dir),
                   enabled_when='handler.state=="running" and '
                                'not handler.pause_requested'),
            Action(name='Resume', action='resume',
                   image=ImageResource('player_fwd', icon_dir),
                   enabled_when='handler.state=="paused"'),
            '-',
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
    raw_channel = Instance('experiment.channel.Channel')

    def log_trial(self, waveforms, fs, **kwargs):
        super(ABRData, self).log_trial(**kwargs)
        if self.waveform_node is None:
            shape = [0] + list(waveforms.shape)
            self.waveform_node = self.fh.create_earray(
                self.store_node, 'waveforms', tables.Float32Atom(), shape)
            self.waveform_node._v_attrs['fs'] = fs
        self.waveform_node.append(waveforms[np.newaxis])


if __name__ == '__main__':
    from cochlear import configure_logging
    from neurogen.calibration import InterpCalibration
    configure_logging()

    mic_file = os.path.join('c:/data/cochlear/calibration/'
                            '150807 - Golay calibration with 377C10.mic')
    c = InterpCalibration.from_mic_file(mic_file)
    log.debug('====================== MAIN =======================')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        paradigm = ABRParadigm(averages=512, reject_threshold=0.2,
                               level='exact_order([40, 60, 80], c=1)',
                               repetition_rate=20,
                               frequency='u(exact_order([4000], c=1), level)')
        experiment = ABRExperiment(paradigm=paradigm, data=data)
        controller = ABRController(mic_cal=c)
        experiment.configure_traits(handler=controller)
