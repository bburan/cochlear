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
from neurogen.calibration.util import psd_freq, psd, rms, tone_power_conv
from experiment.channel import FileChannel

ADC_FS = 200e3


class StandardCalSettings(AbstractParadigm):

    kw = dict(log=True, context=True)
    duration = Float(1, label='Recording duration (sec)', **kw)
    input = Enum('ai0', ('ai0', 'ai1', 'ai2'),
                 label='Analog input (channel)', **kw)
    expected_range = Float(10, label='Expected range (V)', **kw)

    traits_view = View(
        VGroup(
            'duration',
            'input',
            'expected_range',
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
        self.model.component = None
        self.complete = False
        self.state = 'running'
        self.initialize_context()
        self.refresh_context()
        duration = self.get_current_value('duration')
        input = self.get_current_value('input')
        input_line = '/{}/{}'.format(self.DEV, input)
        expected_range = self.get_current_value('expected_range')
        self.iface_adc = ContinuousDAQmxSource(fs=self.adc_fs,
                                               input_line=input_line,
                                               callback=self.poll,
                                               callback_samples=self.adc_fs/8,
                                               expected_range=expected_range)
        self.samples_acquired = 0
        self.target_samples = int(duration*self.adc_fs)
        self.waveforms = []
        self.iface_adc.setup()
        self.iface_adc.start()

    def stop_experiment(self, info=None):
        self.iface_adc.clear()
        self.complete = True

    def poll(self, waveform):
        self.waveforms.append(waveform)
        self.samples_acquired += int(waveform.shape[-1])
        if self.samples_acquired >= self.target_samples:
            self.stop()
            waveforms = np.concatenate(self.waveforms, axis=-1).ravel()
            self.model.generate_plots(waveforms[..., :self.target_samples],
                                      self.adc_fs)


class StandardCal(HasTraits):

    paradigm = Instance(StandardCalSettings, ())
    data = Instance(AbstractData, ())
    component = Instance('enable.api.Component')

    rms = Float(0, label='Overall RMS (mV)')
    psd_rms = Float(0, label='PSD rms (mV)')
    psd_rms_db = Float(0, label='PSD rms (dB re mV)')

    def generate_plots(self, waveform, fs):
        container = VPlotContainer(padding=70, spacing=70)
        frequencies = psd_freq(waveform, fs)
        psd_hanning = psd(waveform, fs, 'hanning')
        self.rms = 1e3*rms(waveform, detrend=True)
        self.psd_rms = psd_hanning[frequencies >= 500].mean()
        self.psd_rms_db = db(psd_hanning[frequencies >= 500], 1e-3).mean()


        time = np.arange(len(waveform))/fs
        plot = create_line_plot((time, waveform*1e3), color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Mic. signal (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (sec)")
        plot.underlays.append(axis)
        container.insert(0, plot)
        subcontainer = OverlayPlotContainer()

        plot = create_line_plot((frequencies, db(psd_hanning, 1e-3)),
                                color='black')
        index_range = DataRange1D(low_setting=10, high_setting=100e3)
        index_mapper = LogMapper(range=index_range)
        plot.index_mapper = index_mapper

        axis = PlotAxis(component=plot, orientation='left',
                        title="Mic. spectrum (dB re mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
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
                Item('psd_rms', style='readonly'),
                Item('psd_rms_db', style='readonly'),
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
        handler = StandardCalController()
        StandardCal().edit_traits(handler=handler, **kwargs)


if __name__ == '__main__':
    handler = StandardCalController()
    StandardCal().configure_traits(handler=handler)
