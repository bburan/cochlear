import numpy as np

from traits.api import (HasTraits, Instance, Float, Int, Property,
                        cached_property)
from traitsui.api import View, HSplit, ToolBar, Action, Item, VGroup
from pyface.api import ImageResource
from enable.api import ComponentEditor
from chaco.api import (create_line_plot, PlotAxis, VPlotContainer, DataRange1D,
                       LogMapper, OverlayPlotContainer)

from cochlear import nidaqmx as ni
from experiment import (AbstractController, icon_dir, AbstractParadigm,
                        AbstractData)
from neurogen.util import db, dbtopa
from neurogen.calibration.util import psd_freq, psd, rms, tone_power_conv

ADC_FS = 200e3


class StandardCalSettings(AbstractParadigm):

    kw = dict(log=True, context=True)
    duration = Float(1, label='Recording duration (sec)', **kw)
    frequency = Float(1000, label='Standard frequency (Hz)', **kw)
    level = Float(114, label='Standard level (dB SPL)', **kw)

    traits_view = View(
        VGroup(
            'duration',
            'frequency',
            'level',
        ),
    )


class StandardCalController(AbstractController):

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
        input_line = ni.DAQmxDefaults.REF_MIC_INPUT
        self.iface_adc = ni.DAQmxInput(fs=self.adc_fs,
                                       input_line=input_line,
                                       callback=self.poll,
                                       callback_samples=int(self.adc_fs/8),
                                       expected_range=10)
        self.samples_acquired = 0
        self.target_samples = int(duration*self.adc_fs)
        self.waveforms = []
        self.iface_adc.start()

    def stop_experiment(self, info=None):
        self.iface_adc.clear()
        self.complete = True

    def poll(self, waveform):
        self.waveforms.append(waveform)
        self.samples_acquired += int(waveform.shape[-1])
        if self.samples_acquired >= self.target_samples:
            self.stop()
            waveforms = np.concatenate(self.waveforms, axis=-1)
            self.model.generate_plots(waveforms.ravel(),
                                      self.adc_fs,
                                      self.get_current_value('frequency'),
                                      self.get_current_value('level'))


class StandardCal(HasTraits):

    paradigm = Instance(StandardCalSettings, ())
    data = Instance(AbstractData, ())
    component = Instance('enable.api.Component')

    # Calculated statistics
    rms = Float(0, label='Overall RMS (mV)')
    peak_freq = Float(0, label='Actual freq. (Hz)')
    rms_freq = Float(0, label='RMS at nom. freq. (mV)')
    mic_sens = Float(0, label='Mic. sens. at nom. freq. (mV/Pa)')
    rms_peak_freq = Float(0, label='RMS at actual freq. (mV)')
    mic_sens_peak_freq = Float(0, label='Mic. sens. at actual freq. (mV/Pa)')

    def generate_plots(self, waveform, fs, frequency, level):
        container = VPlotContainer(padding=70, spacing=70)

        pa = dbtopa(level)

        self.rms = 1e3*rms(waveform, detrend=True)
        self.rms_freq = 1e3*tone_power_conv(waveform, fs, frequency, 'flattop')

        frequencies = psd_freq(waveform, fs)
        psd_hanning = psd(waveform, fs, 'hanning')
        psd_flattop = psd(waveform, fs, 'flattop')

        freq_lb, freq_ub = frequency*0.9, frequency*1.1
        mask = (frequencies >= freq_lb) & (frequencies < freq_ub)
        self.peak_freq = frequencies[mask][np.argmax(psd_hanning[mask])]
        self.rms_peak_freq = 1e3*tone_power_conv(waveform, fs, self.peak_freq,
                                                 'flattop')
        self.mic_sens_peak_freq = self.rms_peak_freq/pa
        self.mic_sens = self.rms_freq/pa

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
                Item('mic_sens_peak_freq', style='readonly'),
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
