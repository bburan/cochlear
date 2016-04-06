import logging
log = logging.getLogger(__name__)

from collections import defaultdict

from traits.api import (Instance, Float, push_exception_handler, Bool,
                        HasTraits, Str, List, Dict, Button, Event)
from traitsui.api import (View, Item, ToolBar, Action, VGroup, HSplit, Tabbed,
                          Include, ListEditor, ListStrEditor)
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (LinearMapper, DataRange1D, PlotAxis, VPlotContainer,
                       OverlayPlotContainer, create_line_plot, ArrayPlotData,
                       Plot)

import numpy as np
from scipy import signal

from cochlear import calibration

from experiment import (AbstractParadigm, AbstractData, AbstractController,
                        AbstractExperiment)
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


################################################################################
# GUI class
################################################################################
class ABRParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    # Signal acquisition settings
    averages = Float(512, dtype=np.int, **kw)
    window = Float(8.5e-3, dtype=np.float, **kw)
    reject_threshold = Float(0.2, dtype=np.float, **kw)
    exp_mic_gain = Float(40, dtype=np.float, **kw)

    # Stimulus settings
    repetition_rate = Float(20, dtype=np.float, **kw)

    duration = Float(5e-3, dtype=np.float, **kw)
    ramp_duration = Float(0.5e-3, dtype=np.float, **kw)

    frequencies = List([8000, 4000, 2000], **kw)
    levels = List([60, 80], **kw)

    add_frequency = Button('+')
    add_level = Button('+')

    def _add_frequency_fired(self):
        self.frequencies.append('')

    def _add_level_fired(self):
        self.levels.append('')

    traits_view = View(
        VGroup(
            VGroup(
                'averages',
                'window',
                label='Acquisition settings',
                show_border=True,
            ),
            VGroup(
                VGroup(
                    Item('frequencies',
                         editor=ListStrEditor(auto_add=True, editable=True)),
                    'add_frequency',
                    show_labels=False
                ),
                VGroup(
                    Item('levels',
                         editor=ListStrEditor(auto_add=True, editable=True)),
                    'add_level',
                    show_labels=False
                ),
                'duration',
                'ramp_duration',
                'repetition_rate',
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

    adc_fs = 100e3
    dac_fs = 100e3

    epoch_info = List()
    epoch_info_lookup = Dict()
    epoch_info_updated = Event()

    def next_trial(self, info=None):
        frequencies = [float(f) for f in self.model.paradigm.frequencies]
        levels = [float(l) for l in self.model.paradigm.levels]
        window = self.model.paradigm.window
        averages = self.model.paradigm.averages
        reject_threshold = self.model.paradigm.reject_threshold
        repetition_rate = self.model.paradigm.repetition_rate
        pip_averages = int(np.ceil(averages/2.0))

        self.model.initialize(frequencies, levels, pip_averages, window,
                              self.adc_fs)

        # Setup the calibration and calibrate each frequency
        self.mic_cal.set_fixed_gain(-self.model.paradigm.exp_mic_gain)
        output_calibration = calibration.multitone_calibration(
            frequencies, self.mic_cal, gain=-20.0)

        self.acq = ABRAcquisition(
            frequencies=frequencies,
            levels=levels,
            calibration=output_calibration,
            pip_averages=pip_averages,
            window=window,
            reject_threshold=reject_threshold,
            repetition_rate=repetition_rate,
            samples_acquired_callback=self.samples_acquired,
            valid_epoch_callback=self.valid_epoch,
            invalid_epoch_callback=self.invalid_epoch,
            done_callback=self.acquisition_done,
            adc_fs=self.adc_fs,
            dac_fs=self.dac_fs,
        )

        for frequency in frequencies:
            for level in levels:
                ei = dict(frequency=frequency, level=level, valid=0, invalid=0)
                self.epoch_info.append(ei)
                self.epoch_info_lookup[frequency, level] = ei

        self.acq.start()

    def acquisition_done(self, state):
        self.stop()

    def request_stop(self, info=None):
        self.acq.stop()
        self.acq.join()

    def stop_experiment(self, info=None):
        pass

    def valid_epoch(self, frequency, level, polarity, presentation, epoch):
        self.epoch_info_lookup[frequency, level]['valid'] += 1
        self.model.save_epoch(frequency, level, polarity, presentation, epoch)
        self.epoch_info_updated = True

    def invalid_epoch(self, level, frequency, polarity, epoch):
        self.epoch_info_lookup[frequency, level]['invalid'] += 1
        self.epoch_info_updated = True

    def samples_acquired(self, samples):
        self.model.data.signal.send(samples.ravel())


class ABRPlot(HasTraits):

    parameter = Str
    stack = Instance(Component)

    def _stack_default(self):
        return VPlotContainer(padding=40, spacing=10, bgcolor='transparent')

    traits_view = View(
        Item('stack', show_label=False,
             editor=ComponentEditor(width=300, height=300))
    )


class ABRData(AbstractData):

    # Store the raw signal from the experiment
    signal = Instance('experiment.channel.Channel')
    waveforms = Instance('tables.Array')

    def initialize_signal(self, adc_fs):
        self.signal = FileChannel(node=self.store_node, fs=adc_fs,
                                  name='signal')

    def initialize_waveforms(self, shape):
        self.waveforms = self.fh.create_array(self.store_node, 'waveforms',
                                              None, atom=tables.Float32Atom(),
                                              shape=shape)


from traitsui.api import VGroup, Item, TabularEditor
from traitsui.tabular_adapter import TabularAdapter

class EpochAdapter(TabularAdapter):

    columns = ['frequency', 'level', 'valid', 'invalid']

    def get_content(self, object, trait, row, column):
        return getattr(object, trait)[row][self.columns[column]]


epoch_editor = TabularEditor(adapter=EpochAdapter(), editable=False,
                             update='handler.epoch_info_updated')


class ABRExperiment(AbstractExperiment):

    paradigm = Instance(ABRParadigm, ())
    data = Instance(AbstractData, ())
    initialized = Bool(False)

    datasource = Instance(ArrayPlotData)
    running_plot = Instance('enable.api.Component')
    abr_waveforms = List(Instance(ABRPlot))

    index_map = Dict()

    def save_epoch(self, frequency, level, phase, presentation, epoch):
        i = self.index_map[frequency, level]
        pi = 0 if phase == 0 else 1
        self.data.waveforms[i, presentation, pi] = epoch

        if (presentation % 8) == 0:
            waveforms = self.data.waveforms[i, :presentation, :]
            waveform = waveforms.mean(axis=0).mean(axis=0)
            key = '{}:{}'.format(frequency, level)
            self.datasource.set_data(key, waveform)

    def initialize(self, frequencies, levels, pip_averages, window, fs):
        # Setup the data
        n_trials = int(len(frequencies)*len(levels))
        window_samples = (window*fs)
        shape = (n_trials, pip_averages, 2, window_samples)
        self.data.initialize_waveforms(shape)
        self.data.initialize_signal(fs)
        self.create_raw_channel_plot(self.data.signal)
        t = np.arange(window_samples, dtype=np.float32)/fs
        self.create_abr_plots(t, frequencies, levels)

    def create_abr_plots(self, t, frequencies, levels):
        i = 0
        self.datasource = ArrayPlotData(time=t*1e3)
        plot_containers = []
        for frequency in frequencies:
            plot_container = ABRPlot(parameter=str(frequency))
            for level in levels:
                self.index_map[frequency, level] = i
                key = '{}:{}'.format(frequency, level)
                self.datasource.set_data(key, [])
                plot = Plot(self.datasource, padding=5, spacing=5)
                plot.plot(('time', key), type='line', color='black')
                plot.underlays = []
                axis = PlotAxis(orientation='left', component=plot,
                                title='Signal (mv)')
                plot.underlays.append(axis)
                plot_container.stack.add(plot)
                i += 1
            plot_containers.append(plot_container)
        self.abr_waveforms = plot_containers

    def create_raw_channel_plot(self, signal):
        container = OverlayPlotContainer(padding=[75, 20, 20, 50])
        data_range = ChannelDataRange(sources=[signal], span=1, trig_delay=.1)
        index_mapper = LinearMapper(range=data_range)
        value_range = DataRange1D(low_setting=-.001, high_setting=.001)
        value_mapper = LinearMapper(range=value_range)
        plot = ExtremesChannelPlot(source=signal, index_mapper=index_mapper,
                                   value_mapper=value_mapper)
        tool = ChannelRangeTool(plot)
        plot.tools.append(tool)
        axis = PlotAxis(component=plot, orientation='left', title='Volts (V)')
        plot.underlays.append(axis)
        add_time_axis(plot, 'bottom', fraction=True)
        container.add(plot)
        self.running_plot = container

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
                    Item('handler.epoch_info', editor=epoch_editor,
                         show_label=False),
                ),
                VGroup(
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


if __name__ == '__main__':
    from neurogen.calibration import InterpCalibration
    import os.path

    mic_file = os.path.join('c:/data/cochlear/calibration',
                            '150807 - Golay calibration with 377C10.mic')
    c = InterpCalibration.from_mic_file(mic_file)
    log.debug('====================== MAIN =======================')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ABRData(store_node=fh.root)
        paradigm = ABRParadigm(averages=256, reject_threshold=np.inf,
                               exp_mic_gain=40, repetition_rate=40,
                               frequencies=['1420'], levels=[80])
        experiment = ABRExperiment(paradigm=paradigm, data=data)
        controller = ABRController(mic_cal=c)
        experiment.configure_traits(handler=controller)
