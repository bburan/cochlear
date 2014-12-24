from __future__ import division

import numpy as np
from scipy import signal

from traits.api import Instance, Float, Property, Int
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, HGroup)
from chaco.api import Plot, ArrayPlotData
from enable.api import Component, ComponentEditor
from pyface.api import ImageResource

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, icon_dir)
from experiment.channel import FileChannel
from experiment.coroutine import blocked, rms

from neurogen.block_definitions import (BandlimitedNoise, Cos2Envelope)
from neurogen.calibration import InterpCalibration
from neurogen.calibration.util import (psd, psd_freq, tone_power_conv_nf)
from neurogen.util import db

from nidaqmx import (DAQmxDefaults, DAQmxChannel, ContinuousDAQmxPlayer,
                     DAQmxAttenControl, ContinuousDAQmxSource)

DAC_FS = 100e3
ADC_FS = 100e3

class NoiseExposureData(AbstractData):

    noise_channel = Instance('experiment.channel.Channel')

    def _noise_channel_default(self):
        return FileChannel(node=self.store_node, name='mic_input',
                           expected_duration=60*60*2, dtype=np.float32)


class NoiseExposureParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)
    center_frequency = \
        Expression(6e3, label='Center frequency (Hz)', dtype=np.float, **kw)
    bandwidth = Expression(4e3, label='Bandwidth (Hz)', dtype=np.float, **kw)
    rs = Expression(85, label='Min. atten. in stop band (dB)',
                    dtype=np.float, **kw)
    rp = Expression(0.3, label='Max. ripple in pass band (dB)',
                    dtype=np.float, **kw)
    order = Expression(7, label='Filter order', dtype=np.float, **kw)

    level = Expression(80, label='Level (dB SPL)', dtype=np.float, **kw)
    seed = Expression(1, label='Noise seed', dtype=np.int, **kw)
    duration = Expression(60, label='Exposure duration (sec)',
                          dtype=np.float, **kw)
    rise_time = Expression(10, label='Noise rise time (sec)',
                           dtype=np.float, **kw)

    mic_sens = Float(2.33, label='Mic. sens. (mV/Pa)', dtype=np.float, **kw)
    mic_sens_dbv = Property(depends_on='mic_sens', dtype=np.float,
                            label='Mic. sens. dB(V/Pa)', **kw)
    speaker_sens = Float(50.0, label='Speaker sens. (mV/Pa)', dtype=np.float,
                         **kw)
    speaker_sens_dbv = Property(depends_on='speaker_sens', dtype=np.float,
                                label='Speaker sens. dB(V/Pa)', **kw)

    def _get_mic_sens_dbv(self):
        return db(self.mic_sens*1e-3)

    def _get_speaker_sens_dbv(self):
        return db(self.speaker_sens*1e-3)

    traits_view = View(
        VGroup(
            VGroup(
                VGroup(
                    'center_frequency',
                    'bandwidth',
                    'rp',
                    'rs',
                    'order',
                    label='Filter settings',
                    show_border=True
                ),
                'level',
                'seed',
                'duration',
                'rise_time',
                label='Stimulus',
                show_border=True
            ),
            HGroup(
                VGroup('mic_sens', 'speaker_sens'),
                VGroup('mic_sens_dbv', 'speaker_sens_dbv', style='readonly'),
                label='Hardware settings',
                show_border=True
            ),
        )
    )


class NoiseExposureController(AbstractController, DAQmxDefaults):

    mic_cal = Instance('neurogen.calibration.InterpCalibration')
    poll_rate = 1

    def setup_experiment(self, info=None):
        calibration = InterpCalibration.as_attenuation()
        token = BandlimitedNoise(name='noise') >> Cos2Envelope(name='envelope')
        channel = DAQmxChannel(token=token, voltage_min=-10, voltage_max=10)
        iface_dac = ContinuousDAQmxPlayer(fs=DAC_FS, done_callback=self.stop)
        iface_dac.add_channel(channel, name='primary')
        iface_adc = ContinuousDAQmxSource(
            fs=ADC_FS, pipeline=blocked(int(ADC_FS*self.poll_rate), -1, self),
            callback_samples=25e3)
        self.channel = channel
        self.iface_adc = iface_adc
        self.iface_dac = iface_dac
        self.token = token

    def send(self, data):
        self.model.update_plots(ADC_FS, data)
        self.model.data.noise_channel.send(data)

    def start_experiment(self, info=None):
        self.register_dtypes()
        self.refresh_context(evaluate=True)
        self.iface_adc.start()
        self.iface_dac.play_continuous()
        self.log_trial()

    def stop_experiment(self, info=None):
        self.state = 'halted'
        self.iface_adc.stop()
        self.iface_dac.stop()

    def set_duration(self, value):
        self.iface_dac.set_value('primary.envelope.duration', value)
        self.iface_dac.duration = value
        self.model.overall_rms_plot.index_range.high_setting = value

    def set_ramp_duration(self, value):
        self.iface_dac.set_value('primary.envelope.rise_time', value)
        self.iface_dac.duration = value

    def set_center_frequency(self, value):
        self.iface_dac.set_value('primary.noise.fc', value)

    def set_bandwidth(self, value):
        self.iface_dac.set_value('primary.noise.bandwidth', value)

    def set_level(self, value):
        self.iface_dac.set_value('primary.noise.level', value)

    def set_seed(self, value):
        self.iface_dac.set_value('primary.noise.seed', value)

    def set_rise_time(self, value):
        self.iface_dac.set_value('primary.envelope.rise_time', value)

    def set_order(self, value):
        self.iface_dac.set_value('primary.noise.order', value)

    def set_rs(self, value):
        self.iface_dac.set_value('primary.noise.rs', value)

    def set_rp(self, value):
        self.iface_dac.set_value('primary.noise.rp', value)

    def set_speaker_sens_dbv(self, value):
        self.channel.calibration = InterpCalibration([0, 100e3], [value, value])


class NoiseExposureExperiment(AbstractExperiment):

    paradigm = Instance(NoiseExposureParadigm, ())
    data = Instance(AbstractData, ())

    rms_data = Instance(ArrayPlotData)
    recent_rms_plot = Instance(Component)
    overall_rms_plot = Instance(Component)
    fft_plot = Instance(Component)

    current_time = Float(0)
    current_update = Int(0)

    current_spl = Float(np.nan, label='Current inst. output (dB SPL)')
    current_spl_average = Float(np.nan, label='Average of last min. (dB SPL)')
    overall_spl_average = Float(np.nan, label='Average output (dB SPL)')

    _coefs = None
    _zf = None

    def update_plots(self, fs, data):
        self.current_update += 1
        data = signal.detrend(data.ravel())

        # Plot RMS
        if self._coefs is None:
            self._coefs = signal.iirfilter(2, (400.0/(fs/2), 40e3/(fs/2)))
            b, a = self._coefs
            self._zf = signal.lfiltic(b, a, data[:len(a)-1], data[:len(b)-1])
        b, a = self._coefs

        data, self._zf = signal.lfilter(b, a, data, zi=self._zf)
        rms = np.mean(data**2)**0.5
        db_rms = db(rms)-self.paradigm.mic_sens_dbv-db(20e-6)
        self.append_data(time=self.current_time, rms=db_rms)
        self.current_time += len(data)/fs

        self.current_spl = db_rms
        self.current_spl_average = self.rms_data.get_data('rms')[-10:].mean()
        self.overall_spl_average = self.rms_data.get_data('rms').mean()

        w_frequency = psd_freq(data, fs)
        w_psd = psd(data, fs, 'boxcar')
        w_psd_db = db(w_psd)-self.paradigm.mic_sens_dbv-db(20e-6)

        self.rms_data.update_data(frequency=w_frequency, psd=w_psd_db)

    def _rms_data_default(self):
        return ArrayPlotData(time=[], rms=[], frequency=[], psd=[])

    def append_data(self, **kwargs):
        for k, v in kwargs.items():
            kwargs[k] = np.append(self.rms_data.get_data(k), v)
        self.rms_data.update_data(**kwargs)

    def _overall_rms_plot_default(self):
        plot = Plot(self.rms_data)
        #plot.index_range.high_setting = 60*60*2
        plot.index_range.low_setting = 0
        #plot.value_range.low_setting = 70
        #plot.value_range.high_setting = 110
        plot.plot(('time', 'rms'))
        return plot

    def _recent_rms_plot_default(self):
        plot = Plot(self.rms_data)
        plot.index_range.high_setting = 'auto'
        plot.index_range.low_setting = 'track'
        plot.index_range.tracking_amount = 60
        #plot.value_range.low_setting = 70
        #plot.value_range.high_setting = 110
        plot.plot(('time', 'rms'))
        return plot

    def _fft_plot_default(self):
        plot = Plot(self.rms_data)
        plot.index_range.low_setting = 1e3
        plot.index_range.high_setting = 20e3
        plot.value_range.low_setting = 10
        plot.value_range.high_setting = 80
        plot.plot(('frequency', 'psd'))
        plot.index_scale = 'log'
        return plot

    traits_view = View(
        HSplit(
            VGroup(
                VGroup(
                    Item('paradigm', style='custom', show_label=False,
                         width=200),
                    show_border=True,
                    label='Settings'
                ),
                VGroup(
                    'current_spl',
                    'current_spl_average',
                    'overall_spl_average',
                    style='readonly',
                    show_border=True,
                    label='Output',
                ),
            ),
            VGroup(
                HGroup(
                    Item('overall_rms_plot',
                        editor=ComponentEditor(width=200, height=200)),
                    Item('recent_rms_plot',
                        editor=ComponentEditor(width=200, height=200)),
                    show_labels=False,
                ),
                Item('fft_plot', show_label=False,
                    editor=ComponentEditor(width=200, height=200)),
            ),
            show_labels=False,
        ),
        resizable=True,
        toolbar=ToolBar(
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('stop', icon_dir),
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
            Menu(
                ActionGroup(
                    Action(name='Load calibration', action='load_calibration'),
                ),
                ActionGroup(
                    Action(name='Calibrate reference microphone',
                           action='calibrate_reference_microphone'),
                    Action(name='Calibrate microphone',
                           action='calibrate_microphone'),
                    Action(name='Calibrate speaker',
                           action='calibrate_speaker'),
                ),
                name='&Calibration'
            ),

        ),
        width=0.5,
        height=0.5,
        id='lbhb.NoiseExposureExperiment',
    )


if __name__ == '__main__':
    import PyDAQmx as ni
    import warnings
    import tables
    ni.DAQmxResetDevice('Dev1')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with tables.open_file('temp.hdf5', 'w') as fh:
            data = NoiseExposureData(store_node=fh.root)
            controller = NoiseExposureController()
            NoiseExposureExperiment(data=data) \
                .configure_traits(handler=controller)
