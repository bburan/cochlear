import numpy as np

from traits.api import (HasTraits, Instance, Float, Enum, Int, Property,
                        cached_property, Any)
from traitsui.api import View, HSplit, ToolBar, Action, Item, VGroup
from pyface.api import ImageResource
from enable.api import ComponentEditor
from chaco.api import (create_line_plot, PlotAxis, VPlotContainer, DataRange1D,
                       LogMapper, OverlayPlotContainer)

from nidaqmx import DAQmxDefaults, ContinuousDAQmxSource
from experiment import (AbstractController, icon_dir, AbstractData,
                        AbstractParadigm, AbstractData)
from neurogen.util import db, dbtopa
from experiment.channel import FileChannel

ADC_FS = 200e3


class StandardCalData(AbstractData):

    channel = Instance('experiment.channel.Channel')

    def _channel_default(self):
        return FileChannel(fs=ADC_FS, node=self.store_node, name='temp')


class StandardCalSettings(AbstractParadigm):

    kw = dict(log=True, context=True)
    duration = Float(1, label='Recording duration (sec)', **kw)
    input = Enum('ai2', ('ai0', 'ai1', 'ai2'), label='Analog input (channel)',
                 **kw)
    frequency = Float(1000, label='Standard frequency (Hz)', **kw)
    level = Float(114, label='Standard level (dB SPL)', **kw)
    nom_sens = Float(2.0, label='Nom. sens. (mV/Pa)', **kw)

    traits_view = View(
        VGroup(
            'duration',
            'input',
            'frequency',
            'level',
        ),
    )


class StandardCalController(DAQmxDefaults, AbstractController):

    adc_fs = ADC_FS
    samples_acquired = Int(0)
    acquired = Property(Float, depends_on='samples_acquired')

    @cached_property
    def _get_acquired(self):
        return self.samples_acquired/self.adc_fs

    def start(self, info=None):
        self.complete = False
        self.state = 'running'
        self.initialize_context()
        self.refresh_context()
        duration = self.get_current_value('duration')
        input = self.get_current_value('input')
        input_line = '/{}/{}'.format(self.DEV, input)
        self.iface_adc = ContinuousDAQmxSource(fs=self.adc_fs,
                                               input_line=input_line,
                                               callback=self.poll,
                                               callback_samples=self.adc_fs/8,
                                               expected_range=10)
        self.samples_acquired = 0
        self.target_samples = int(duration*self.adc_fs)
        self.model.data.channel.clear()
        self.iface_adc.setup()
        self.iface_adc.start()

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_adc.clear()
        self.complete = True

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.channel.send(waveform)
        print waveform.shape
        self.samples_acquired += int(waveform.shape[-1])
        if self.samples_acquired >= self.target_samples:
            self.model.generate_plots()
            self.stop()


class StandardCal(HasTraits):

    paradigm = Instance(StandardCalSettings, ())
    data = Instance(AbstractData, ())
    component = Instance('enable.api.Component')

    # Calculated statistics
    rms = Float(0, label='Overall RMS (mV)')
    peak_freq = Float(0, label='Actual freq. (Hz)')
    rms_freq = Float(0, label='RMS at nom. freq. (mV)')
    rms_peak_freq = Float(0, label='RMS at actual freq. (mV)')
    mic_sens = Float(0, label='Mic. sens. (mV/Pa)')

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        frequency = self.paradigm.frequency
        level = self.paradigm.level
        pa = dbtopa(level)

        self.rms = 1e3*self.data.channel.get_rms(detrend=True)
        self.rms_freq = 1e3*self.data.channel.get_magnitude(frequency, rms=True,
                                                            window='flattop')

        frequencies = self.data.channel.get_fftfreq()
        psd_hanning = self.data.channel.get_psd(rms=True, window='hanning')
        psd_flattop = self.data.channel.get_psd(rms=True, window='flattop')

        freq_lb, freq_ub = frequency*0.9, frequency*1.1
        mask = (frequencies >= freq_lb) & (frequencies < freq_ub)
        self.peak_freq = frequencies[mask][np.argmax(psd_hanning[mask])]
        self.rms_peak_freq = \
            1e3*self.data.channel.get_magnitude(self.peak_freq, rms=True,
                                                window='flattop')
        self.mic_sens = self.rms_peak_freq/pa

        plot = create_line_plot((self.data.channel.time,
                                 self.data.channel[:]*1e3),
                                color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Mic. signal (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (sec)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        subcontainer = OverlayPlotContainer()

        index_range = DataRange1D(low_setting=freq_lb, high_setting=freq_ub)
        index_mapper = LogMapper(range=index_range)
        plot = create_line_plot((frequencies, db(psd_hanning, 1e-3)),
                                color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Mic. spectrum (dB re mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        subcontainer.add(plot)

        plot = create_line_plot((frequencies, db(psd_flattop, 1e-3)),
                                color='red')
        plot.index_mapper = index_mapper
        subcontainer.add(plot)

        container.insert(0, subcontainer)

        self.component = container

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.acquired', style='readonly',
                     label='Acquired (sec)'),
                Item('rms', style='readonly'),
                Item('peak_freq', style='readonly'),
                Item('rms_freq', style='readonly'),
                Item('rms_peak_freq', style='readonly'),
                Item('mic_sens', style='readonly'),
                VGroup(
                    Item('component', editor=ComponentEditor(),
                         show_label=False),
                ),
            ),
            show_labels=False,
            show_border=True,
        ),
        toolbar=ToolBar(
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state!="running"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.state=="running"'),
        ),
        resizable=True,
    )


def launch_gui(**kwargs):
    import tables
    with tables.open_file('dummy', 'w', driver='H5FD_CORE',
                          core_backing_store=0) as fh:
        data = StandardCalData(store_node=fh.root)
        StandardCal(data=data).configure_traits(handler=StandardCalController())
