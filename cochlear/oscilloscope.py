from __future__ import division
from cns import get_config

from math import ceil
from os import path
from tempfile import mkdtemp

import numpy as np
import tables

from traits.api import (Instance, Any, List, on_trait_change, Enum, Dict,
                        HasTraits, Range, Tuple, Bool, Int, Str, Property,
                        cached_property, Button, Float, Undefined)
from traitsui.api import (Controller, View, VGroup, HGroup, Item, Label,
                          Include, TableEditor, ObjectColumn, InstanceEditor,
                          RangeEditor, HSplit, Tabbed, ShellEditor)
from traitsui.extras.checkbox_column import CheckboxColumn
from enable.api import Component, ComponentEditor
from chaco.api import (PlotAxis, LinearMapper, DataRange1D,
                       OverlayPlotContainer)
from chaco.tools.api import ZoomTool
from pyface.api import confirm, YES
from pyface.timer.api import Timer

from cns.util import to_list

from experiment.plots.tools.window_tool import WindowTool
from experiment.plots.helpers import add_time_axis
from experiment.plots.channel_data_range import ChannelDataRange
from experiment.plots.extremes_multi_channel_plot \
    import ExtremesMultiChannelPlot
from experiment.plots.ttl_plot import TTLPlot
from experiment.plots.epoch_plot import EpochPlot
from experiment.plots.channel_range_tool import MultiChannelRangeTool
from experiment.plots.channel_number_overlay import ChannelNumberOverlay
from experiment.plots.snippet_channel_plot import SnippetChannelPlot

from experiment.plots.spike_overlay import SpikeOverlay
from experiment.plots.threshold_overlay import ThresholdOverlay

from experiment.channel import (FileMultiChannel, FileChannel, FileTimeseries,
                                FileEpoch)
from experiment.channel import FileEpochChannel as FileSnippetChannel

from experiment.util import load_instance, dump_instance

from traitsui.menu import MenuBar, Menu, ActionGroup, Action

CHANNELS = 1
PHYSIOLOGY_WILDCARD = get_config('PHYSIOLOGY_WILDCARD')
SPIKE_SNIPPET_SIZE = get_config('PHYSIOLOGY_SPIKE_SNIPPET_SIZE')
SNIPPET_SIZE = get_config('PHYSIOLOGY_SPIKE_SNIPPET_SIZE')
VOLTAGE_SCALE = 1e3
PHYSIOLOGY_ROOT = 'c:/data'

from nidaqmx import ContinuousDAQmxSource


def scale_formatter(x):
    return "{:.2f}".format(x*VOLTAGE_SCALE)


class PhysiologyController(Controller):

    buffer_ = Any
    iface_physiology = Any
    buffer_raw = Any
    buffer_proc = Any
    buffer_ts = Any
    buffer_ttl = Any
    physiology_ttl_pipeline = Any
    buffer_spikes = List(Any)
    state = Enum('master', 'client')
    timer = Instance(Timer)
    parent = Any

    shell_variables = Dict

    # These define what variables will be available in the Python shell.  Right
    # now we can't add various stuff such as the data and interface classes
    # because they have not been created yet.  I'm not sure how we can update
    # the Python shell with new instances once the experiment has started
    # running.
    def _shell_variables_default(self):
        return dict(controller=self, c=self)

    def init(self, info):
        self.setup_physiology()
        self.model = info.object
        if self.state == 'master':
            self.start()

    def close(self, info, is_ok, from_parent=False):
        return True
        if self.state == 'client':
            return from_parent
        else:
            # The function confirm returns an integer that represents the
            # response that the user requested.  YES is a constant (also
            # imported from the same module as confirm) corresponding to the
            # return value of confirm when the user presses the "yes" button on
            # the dialog.  If any other button (e.g. "no", "abort", etc.) is
            # pressed, the return value will be something other than YES and we
            # will assume that the user has requested not to quit the
            # experiment.
            if confirm(info.ui.control, 'OK to stop?') == YES:
                self.stop(info)
                return True
            else:
                return False

    def setup_physiology(self):
        self.buffer_raw = \
            ContinuousDAQmxSource(input_line='/Dev1/ai0',
                                  callback=self.monitor_physiology,
                                  callback_samples=5e3)
        self.buffer_raw.setup()

    @on_trait_change('model.data')
    def update_data(self):
        # Ensure that the data store has the correct sampling frequency
        #for i in range(CHANNELS):
        #    data_spikes = self.model.data.spikes
        #    for src, dest in zip(self.buffer_spikes, data_spikes):
        #        dest.fs = src.fs
        #        dest.snippet_size = SPIKE_SNIPPET_SIZE
        self.model.data.raw.fs = self.buffer_raw.fs
        self.model.data.processed.fs = self.buffer_raw.fs
        #self.model.data.processed.fs = self.buffer_filt.fs
        #self.model.data.ts.fs = self.iface_physiology.fs
        #self.model.data.epoch.fs = self.iface_physiology.fs
        #self.model.data.sweep.fs = self.buffer_ttl.fs

        # Setup the pipeline
        #targets = [self.model.data.sweep]
        #self.physiology_ttl_pipeline = deinterleave_bits(targets)

    def start(self):
        self.buffer_raw.start()
        #self.timer = Timer(100, self.monitor_physiology)

    def stop(self):
        #self.timer.stop()
        #self.process.stop()
        self.buffer_raw.clear()

    def monitor_physiology(self):
        # Acquire raw physiology data
        waveform = self.buffer_raw.read_analog()
        self.model.data.raw.send(waveform)
        self.model.data.processed.send(waveform)

        # Acquire filtered physiology data
        #waveform = self.buffer_filt.read()
        #self.model.data.processed.send(waveform)

        # Acquire sweep data
        #ttl = self.buffer_ttl.read()
        #self.physiology_ttl_pipeline.send(ttl)

        ## Get the timestamps
        #ts = self.buffer_ts.read().ravel()
        #self.model.data.ts.send(ts)

        #ends = self.buffer_ts_end.read().ravel()
        #starts = self.buffer_ts_start.read(len(ends)).ravel()
        #self.model.data.epoch.send(zip(starts, ends))

    #@on_trait_change('model.settings.spike_signs')
    #def set_spike_signs(self, value):
    #    for ch, sign in enumerate(value):
    #        name = 's_spike{}'.format(ch+1)
    #        self.iface_physiology.set_tag(name, sign)

    #@on_trait_change('model.settings.spike_thresholds')
    #def set_spike_thresholds(self, value):
    #    for ch, threshold in enumerate(value):
    #        name = 'a_spike{}'.format(ch+1)
    #        self.iface_physiology.set_tag(name, threshold)

    #@on_trait_change('model.settings.monitor_fc_highpass')
    #def set_monitor_fc_highpass(self, value):
    #    self.iface_physiology.set_tag('FiltHP', value)

    #@on_trait_change('model.settings.monitor_fc_lowpass')
    #def set_monitor_fc_lowpass(self, value):
    #    self.iface_physiology.set_tag('FiltLP', value)

    #@on_trait_change('model.settings.monitor_ch_1')
    #def set_monitor_ch_1(self, value):
    #    self.iface_physiology.set_tag('ch1_out', value)

    #@on_trait_change('model.settings.monitor_ch_2')
    #def set_monitor_ch_2(self, value):
    #    self.iface_physiology.set_tag('ch2_out', value)

    #@on_trait_change('model.settings.monitor_ch_3')
    #def set_monitor_ch_3(self, value):
    #    self.iface_physiology.set_tag('ch3_out', value)

    #@on_trait_change('model.settings.monitor_gain_1')
    #def set_monitor_gain_1(self, value):
    #    self.iface_physiology.set_tag('ch1_out_sf', value*1e3)

    #@on_trait_change('model.settings.monitor_gain_2')
    #def set_monitor_gain_2(self, value):
    #    self.iface_physiology.set_tag('ch2_out_sf', value*1e3)

    #@on_trait_change('model.settings.monitor_gain_3')
    #def set_monitor_gain_3(self, value):
    #    self.iface_physiology.set_tag('ch3_out_sf', value*1e3)

    #@on_trait_change('model.settings.diff_matrix')
    #def set_diff_matrix(self, value):
    #    self.iface_physiology.set_coefficients('diff_map', value.ravel())

    def load_settings(self, info):
        instance = load_instance(PHYSIOLOGY_ROOT, PHYSIOLOGY_WILDCARD)
        if instance is not None:
            self.model.settings.copy_traits(instance)

    def saveas_settings(self, info):
        dump_instance(self.model.settings, PHYSIOLOGY_ROOT, PHYSIOLOGY_WILDCARD)


def create_menubar():
    actions = ActionGroup(
        Action(name='Load settings', action='load_settings'),
        Action(name='Save settings as', action='saveas_settings'),
        Action(name='Restore defaults', action='reset_settings'),
    )
    menu = Menu(actions, name='&Physiology')
    return MenuBar(menu)


def ptt(event_times, trig_times):
    return np.concatenate([event_times-tt for tt in trig_times])


class Histogram(HasTraits):

    channel = Range(1, CHANNELS, 1)
    spikes = Any
    timeseries = Any
    bin_size = Float(0.25)
    bin_lb = Float(-1)
    bin_ub = Float(5)

    bins = Property(depends_on='bin_+')

    def _get_bins(self):
        return np.arange(self.bin_lb, self.bin_ub, self.bin_size)

    def _get_counts(self):
        spikes = self.spikes[self.channel-1]
        et = spikes.timestamps/spikes.fs
        return np.histogram(ptt(spikes, et), bins=self.bins)[0]

    def _get_histogram(self):
        self.spikes.timestamps[:]


class SortWindow(HasTraits):

    settings = Any
    channels = Any
    channel = Range(1, CHANNELS, 1)

    plot = Instance(SnippetChannelPlot)
    threshold = Property(Float, depends_on='channel')
    sign = Property(Bool, depends_on='channel')
    windows = Property(List(Tuple(Float, Float, Float)), depends_on='channel')
    tool = Instance(WindowTool)

    def _get_sign(self):
        return self.settings[self.channel-1].spike_sign

    def _set_sign(self, value):
        self.settings[self.channel-1].spike_sign = value

    def _get_threshold(self):
        return self.settings[self.channel-1].spike_threshold*VOLTAGE_SCALE

    def _set_threshold(self, value):
        self.settings[self.channel-1].spike_threshold = value/VOLTAGE_SCALE

    def _get_windows(self):
        if self.settings is None:
            return []
        return self.settings[self.channel-1].spike_windows

    def _set_windows(self, value):
        if self.settings is None:
            return
        self.settings[self.channel-1].spike_windows = value

    def _channel_changed(self, new):
        self.plot.channel = self.channels[new-1]

    def _plot_default(self):
        # Create the plot
        index_mapper = LinearMapper(range=DataRange1D(low=0, high=0.0012))
        value_mapper = LinearMapper(range=DataRange1D(low=-0.00025,
                                                      high=0.00025))
        plot = SnippetChannelPlot(history=20,
                                  source=self.channels[self.channel-1],
                                  value_mapper=value_mapper,
                                  index_mapper=index_mapper, bgcolor='white',
                                  padding=[60, 5, 5, 20])
        #add_default_grids(plot, major_index=1e-3, minor_index=1e-4,
                          #major_value=1e-3, minor_value=1e-4)

        # Add the axes labels
        axis = PlotAxis(orientation='left', component=plot,
                        tick_label_formatter=scale_formatter,
                        title='Signal (mV)')
        plot.overlays.append(axis)
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=scale_formatter)
        plot.overlays.append(axis)

        # Add the tools
        zoom = ZoomTool(plot, drag_button=None, axis="value")
        plot.overlays.append(zoom)
        self.tool = WindowTool(component=plot)
        plot.overlays.append(self.tool)

        # Whenever we draw a window, the settings should immediately be updated!
        self.sync_trait('windows', self.tool)
        return plot

    THRESHOLD_EDITOR = RangeEditor(low=-5e-4*VOLTAGE_SCALE,
                                   high=5e-4*VOLTAGE_SCALE)

    traits_view = View(
        VGroup(
            HGroup(
                Item('channel', style='text', show_label=False, width=-25),
                Item('sign', label='Signed?'),
                Item('threshold', editor=THRESHOLD_EDITOR, show_label=False,
                     springy=True),
                ),
            Item('plot', editor=ComponentEditor(width=250, height=250)),
            show_labels=False,
        ),
    )


class PhysiologyData(HasTraits):

    store_node = Any

    ############################################################################
    # Permanent data
    ############################################################################
    # Raw (permanent) physiology data that will be stored in the data file that
    # is retained at the end of the experiment.

    raw = Instance(FileMultiChannel)
    sweep = Instance(FileChannel)
    ts = Instance(FileTimeseries)
    epoch = Instance(FileEpoch)

    def _sweep_default(self):
        return FileChannel(node=self.store_node, name='sweep', dtype=np.bool,
                           use_checksum=True)

    def _raw_default(self):
        return FileMultiChannel(node=self.store_node, channels=CHANNELS,
                                name='raw', dtype=np.float32,
                                compression_type='lzo', compression_level=1,
                                use_shuffle=True, use_checksum=True)

    def _ts_default(self):
        return FileTimeseries(node=self.store_node, name='ts', dtype=np.int32,
                              use_checksum=True)

    def _epoch_default(self):
        return FileEpoch(node=self.store_node, name='epoch', dtype=np.int32)

    ############################################################################
    # TEMPORARY DATA
    ############################################################################
    # Temporary data for plotting.  Stored in a temporary file that will
    # eventually be discarded at the end of the experiment.

    # Where to store the temporary data
    temp_node = Any
    processed = Instance(FileMultiChannel)
    spikes = List(Instance(FileSnippetChannel))

    def _temp_node_default(self):
        filename = path.join(mkdtemp(), 'processed_physiology.h5')
        tempfile = tables.openFile(filename, 'w')
        return tempfile.root

    def _processed_default(self):
        return FileMultiChannel(node=self.temp_node, channels=CHANNELS,
                                name='processed', dtype=np.float32)

    def _spikes_default(self):
        channels = []
        for i in range(CHANNELS):
            name = 'spike_{:02}'.format(i+1)
            ch = FileSnippetChannel(node=self.temp_node, name=name,
                                    dtype=np.float32, snippet_size=SNIPPET_SIZE)
            channels.append(ch)
        return channels


class ChannelSetting(HasTraits):

    number = Int
    differential = Str
    visible = Bool(True)
    bad = Bool(False)

    # Threshold for candidate spike used in on-line spike sorting
    spike_threshold = Float(0.0005)
    # Should spike_threshold trigger on +/-?
    spike_sign = Bool(False)

    # Windows used for candidate spike isolation and sorting.
    spike_windows = List(Tuple(Float, Float, Float), [])
    sort_summary = Property(Str, depends_on='spike_+')

    def _get_sort_summary(self):
        if not self.spike_sign:
            t = u"\u00B1{} with {} windows"
        else:
            t = "{:+} with {} windows"
        return t.format(self.spike_threshold, len(self.spike_windows))


channel_editor = TableEditor(
    show_row_labels=True,
    sortable=False,
    reorderable=True,
    columns=[
        CheckboxColumn(name='visible', width=10, label='V'),
        CheckboxColumn(name='bad', width=10, label='B'),
        ObjectColumn(name='differential', width=100),
        ]
)


class PhysiologyParadigm(HasTraits):

    # Width of buttons in GUI
    WIDTH = -40

    # These are convenience buttons for quickly setting the differentials of
    # each channel.
    diff_none = Button('None')
    diff_all = Button('All')

    def _get_diff_group(self, channel, group):
        ub = int(ceil(channel/group)*group + 1)
        lb = ub - group
        diff = range(lb, ub)
        diff.remove(channel)
        diff = [d for d in diff if not self.channel_settings[d-1].bad]
        return ', '.join(str(ch) for ch in diff)

    def _diff_none_fired(self):
        for channel in self.channel_settings:
            channel.set(True, differential='')

    def _diff_all_fired(self):
        for channel in self.channel_settings:
            value = self._get_diff_group(channel.number, CHANNELS)
            channel.set(True, differential=value)

    diff_group = HGroup(
        Item('diff_all', width=WIDTH),
        Item('diff_none', width=WIDTH),
        show_labels=False,
        show_border=True,
        label='Quick differential'
    )

    show_all = Button(label='All')
    show_none = Button(label='None')

    visible_group = HGroup(
        Item('show_all', width=WIDTH),
        Item('show_none', width=WIDTH),
        show_labels=False,
        show_border=True,
        label='Quick visible'
    )

    def _set_visible(self, channels):
        for ch in self.channel_settings:
            ch.visible = ch.number in channels and not ch.bad

    def _all_fired(self):
        self._set_visible(range(1, 17))

    def _none_fired(self):
        self._set_visible([])

    # The RZ5 has four DAC channels.  We can send each of these channels to an
    # oscilloscope for monitoring.  The first DAC channel (corresponding to
    # channel 9 in the RPvds file) is linked to the speaker.  Gain is multiplied
    # by a factor of 1000 before being applied to the corresponding channel.
    monitor_ch_1 = Range(1, CHANNELS, 1)
    monitor_ch_2 = Range(1, CHANNELS, 5)
    monitor_ch_3 = Range(1, CHANNELS, 9)
    monitor_gain_1 = Range(0, 100, 50)
    monitor_gain_2 = Range(0, 100, 50)
    monitor_gain_3 = Range(0, 100, 50)

    # Bandpass filter settings
    monitor_fc_highpass = Range(0, 1e3, 300)
    monitor_fc_lowpass = Range(1e3, 5e3, 5e3)

    channel_settings = List(Instance(ChannelSetting))

    def _channel_settings_default(self):
        return [ChannelSetting(number=i) for i in range(1, CHANNELS+1)]

    # List of the channels visible in the plot
    visible_channels = Property(List, depends_on='channel_settings.visible')

    @cached_property
    def _get_visible_channels(self):
        settings = self.channel_settings
        return [i for i, ch in enumerate(settings) if ch.visible]

    spike_thresholds = Property(depends_on='channel_settings.spike_threshold')

    def _get_spike_thresholds(self):
        return [ch.spike_threshold for ch in self.channel_settings]

    spike_signs = Property(depends_on='channel_settings.spike_sign')

    def _get_spike_signs(self):
        return [ch.spike_sign for ch in self.channel_settings]

    # Generates the matrix that will be used to compute the differential for the
    # channels. This matrix will be uploaded to the RZ5.
    diff_matrix = Property(depends_on='channel_settings.differential')

    @cached_property
    def _get_diff_matrix(self):
        n_chan = len(self.channel_settings)
        map = np.zeros((n_chan, n_chan))
        for channel in self.channel_settings:
            diff = to_list(channel.differential)
            if len(diff) != 0:
                sf = -1.0/len(diff)
                for d in diff:
                    map[channel.number-1, d-1] = sf
            map[channel.number-1, channel.number-1] = 1
        return map

    monitor_group = HGroup(
        VGroup(
            Label('Channel'),
            Item('monitor_ch_1'),
            Item('monitor_ch_2'),
            Item('monitor_ch_3'),
            show_labels=False,
            ),
        VGroup(
            Label('Gain (1000x)'),
            Item('monitor_gain_1'),
            Item('monitor_gain_2'),
            Item('monitor_gain_3'),
            show_labels=False,
            ),
        label='Monitor Settings',
        show_border=True,
    )

    filter_group = HGroup(
        Item('monitor_fc_highpass', style='text'),
        Label('to'),
        Item('monitor_fc_lowpass', style='text'),
        label='Filter Settings',
        show_labels=False,
        show_border=True,
    )

    physiology_view = View(
        VGroup(
            Include('filter_group'),
            Include('monitor_group'),
            HGroup(
                Include('visible_group'),
                Include('diff_group'),
            ),
            Item('channel_settings', editor=channel_editor),
            show_labels=False,
            ),
    )


class PhysiologyExperiment(HasTraits):

    settings = Instance(PhysiologyParadigm, ())
    data = Instance(PhysiologyData)

    physiology_container = Instance(Component)
    physiology_plot = Instance(Component)
    physiology_index_range = Instance(ChannelDataRange)
    physiology_value_range = Instance(DataRange1D, ())
    channel_span = Float(0.5e-3)

    sort_window_1 = Instance(SortWindow)
    sort_window_2 = Instance(SortWindow)
    sort_window_3 = Instance(SortWindow)
    #channel_sort = Property(depends_on='sort_window_+.channel')

    channel = Enum('processed', 'raw')

    # Overlays
    spike_overlay = Instance(SpikeOverlay)
    threshold_overlay = Instance(ThresholdOverlay)
    parent = Any

    # Show the overlays?
    visualize_spikes = Bool(False)
    visualize_thresholds = Bool(False)
    show_channel_number = Bool(True)

    #@cached_property
    #def _get_channel_sort(self):
    #    channels = []
    #    for i in range(min(3, CHANNELS)):
    #        window = getattr(self, 'sort_window_{}'.format(i+1))
    #        # The GUI representation starts at 1, the program representation
    #        # starts at 0.  For 16 channels, the GUI displays the numbers 1
    #        # through 16 which corresponds to 0 through 15 in the code.  We need
    #        # to convert back and forth as needed.
    #        channels.append(window.channel-1)
    #    return channels

    #@on_trait_change('data, settings.channel_settings')
    #def _physiology_sort_plots(self):
    #    settings = self.settings.channel_settings
    #    channels = self.data.spikes
    #    window = SortWindow(channel=1, settings=settings, channels=channels)
    #    self.sort_window_1 = window
    #    window = SortWindow(channel=5, settings=settings, channels=channels)
    #    self.sort_window_2 = window
    #    window = SortWindow(channel=9, settings=settings, channels=channels)
    #    self.sort_window_3 = window

    def _channel_changed(self, new):
        if new == 'raw':
            self.physiology_plot.channel = self.data.raw
        else:
            self.physiology_plot.channel = self.data.processed

    @on_trait_change('data, parent')
    def _generate_physiology_plot(self):
        # NOTE THAT ORDER IS IMPORTANT.  First plots added are at bottom of
        # z-stack, so the physiology must be last so it appears on top.

        # Padding is in left, right, top, bottom order
        container = OverlayPlotContainer(padding=[50, 20, 20, 50])

        # Create the index range shared by all the plot components
        self.physiology_index_range = \
            ChannelDataRange(span=5, trig_delay=1, timeseries=self.data.ts,
                             sources=[self.data.processed])

        # Create the TTL plot
        index_mapper = LinearMapper(range=self.physiology_index_range)
        value_mapper = LinearMapper(range=DataRange1D(low=0, high=1))
        plot = TTLPlot(source=self.data.sweep, index_mapper=index_mapper,
                       value_mapper=value_mapper, reference=0,
                       fill_color=(0.25, 0.41, 0.88, 0.1),
                       line_color='transparent', rect_center=0.5,
                       rect_height=1.0)
        container.add(plot)

        # Create the epoch plot
        plot = EpochPlot(source=self.data.epoch, marker='diamond',
                         marker_color=(.5, .5, .5, 1.0), marker_height=0.9,
                         marker_size=10, index_mapper=index_mapper,
                         value_mapper=value_mapper)
        container.add(plot)

        #add_default_grids(plot, major_index=1, minor_index=0.25)

        # Hack alert.  Can we separate this out into a separate function?
        if self.parent is not None:
            try:
                self.parent._add_experiment_plots(index_mapper, container)
            except AttributeError:
                pass

        # Create the neural plots
        value_mapper = LinearMapper(range=self.physiology_value_range)
        plot = ExtremesMultiChannelPlot(source=self.data.processed,
                                        index_mapper=index_mapper,
                                        value_mapper=value_mapper)
        self.settings.sync_trait('visible_channels', plot, 'channel_visible',
                                 mutual=False)

        overlay = ChannelNumberOverlay(plot=plot)
        self.sync_trait('show_channel_number', overlay, 'visible')
        plot.overlays.append(overlay)

        container.add(plot)
        #add_default_grids(plot, major_index=1, minor_index=0.25,
        #                  major_value=1e-3, minor_value=1e-4)
        axis = PlotAxis(component=plot, orientation='left',
                        tick_label_formatter=scale_formatter,
                        title='Volts (mV)')
        plot.underlays.append(axis)
        add_time_axis(plot, 'bottom', fraction=True)
        self.physiology_plot = plot

        tool = MultiChannelRangeTool(component=plot)
        plot.tools.append(tool)

        #overlay = SpikeOverlay(plot=plot, spikes=self.data.spikes)
        #self.sync_trait('visualize_spikes', overlay, 'visible')
        #plot.overlays.append(overlay)
        #self.spike_overlay = overlay

        # Create the threshold overlay plot
        #overlay = ThresholdOverlay(plot=plot, visible=False)
        #self.sync_trait('visualize_thresholds', overlay, 'visible')
        #self.settings.sync_trait('spike_thresholds', overlay, 'sort_thresholds',
        #                         mutual=False)
        #self.settings.sync_trait('spike_signs', overlay, 'sort_signs',
        #                         mutual=False)
        #self.sync_trait('channel_sort', overlay, 'sort_channels', mutual=False)
        #plot.overlays.append(overlay)
        #self.threshold_overlay = overlay

        self.physiology_container = container

    #zero_delay = Button('Reset trigger delay')
    pause_update = Button('Pause update')
    resume_update = Button('Resume update')
    update_mode = Enum('continuous', 'paused')

    #def _zero_delay_fired(self):
    #    self.physiology_index_range.trig_delay = 0

    def _pause_update_fired(self):
        self.update_mode = 'paused'

    def _resume_update_fired(self):
        self.update_mode = 'continuous'
        if self.parent is not None:
            trial = self.parent.data.trial_log[-1]
            self.physiology_index_range.trigger = trial['start']

    @on_trait_change('parent.selected_trial')
    def _update_selected_trigger(self, row):
        if row is not Undefined:
            # The trial_log widget reverses the rows so the most recent trial is
            # row 0.  We can use negative indexing to access the desired trial.
            # e.g. if it is row 0, then we want the very last trial (e.g. -1).
            # If it is row 5, then this means that we want the sixth-to-last
            # trial (e.g. row -6)).
            start = self.parent.data.trial_log[row]['start']
            self.physiology_index_range.update_mode = 'triggered'
            self.physiology_index_range.trigger = start
            self.update_mode = 'paused'

    @on_trait_change('data:ts.added')
    def _update_trigger(self, timestamps):
        if timestamps is not Undefined and self.update_mode == 'continuous':
            self.physiology_index_range.trigger = timestamps[-1]

    trigger_buttons = HGroup(
        'pause_update',
        'resume_update',
        show_labels=False,
    )

    physiology_settings_group = VGroup(
        HGroup(
            Item('show_channel_number', label='Show channel number'),
            Item('channel'),
            ),
        Item('settings', style='custom',
             editor=InstanceEditor(view='physiology_view')),
        Include('physiology_view_settings_group'),
        show_border=True,
        show_labels=False,
        label='Channel settings'
    )

    physiology_view_settings_group = VGroup(
        Include('trigger_buttons'),
        Item('object.physiology_index_range.update_mode', label='Trigger mode'),
        Item('object.physiology_index_range.span', label='X span',
             editor=RangeEditor(low=0.1, high=30.0)),
        Item('channel_span', label='Y span',
             editor=RangeEditor(low=0, high=10e3),
             width=100),
        Item('object.physiology_index_range.trig_delay', label='Trigger delay'),
        label='Plot Settings',
        show_border=True,
    )

    sort_settings_group = VGroup(
        HGroup(
            Item('visualize_spikes', label='Show sorted spikes?'),
            Item('visualize_thresholds', label='Show sort threshold?'),
            show_border=True,
            ),
        Item('sort_window_1', style='custom', width=250),
        Item('sort_window_2', style='custom', width=250),
        Item('sort_window_3', style='custom', width=250),
        show_labels=False,
        label='Sort settings'
    )

    physiology_view = View(
        HSplit(
            Tabbed(
                Include('physiology_settings_group'),
                VGroup(
                    Item('object.threshold_overlay', style='custom'),
                    Item('object.spike_overlay', style='custom'),
                    Item('object.physiology_plot', style='custom'),
                    show_labels=False,
                    label='GUI settings'
                    ),
            ),
            Tabbed(
                Item('physiology_container',
                     editor=ComponentEditor(width=500, height=800),
                     width=500, resizable=True),
                Item('handler.shell_variables', editor=ShellEditor()),
                show_labels=False,
            ),
            show_labels=False,
            ),
        menubar=create_menubar(),
        handler=PhysiologyController,
        resizable=True,
        height=0.95,
        width=0.95,
    )


if __name__ == '__main__':
    import os

    # Create a temporary file (we can't create a temporary file using the
    # standard tempfile functions tables.openFile requires a string, not a
    # file-like object).
    tempfile = os.path.join(get_config('TEMP_ROOT'), 'test_physiology.h5')

    # Run the experiment
    with tables.openFile(tempfile, 'w') as datafile:
        data = PhysiologyData(store_node=datafile.root)
        controller = PhysiologyController()
        PhysiologyExperiment(data=data).configure_traits(handler=controller)
        #PhysiologyExperiment(data=data).configure_traits()

    # Delete the temporary file
    os.unlink(tempfile)
