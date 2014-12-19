from traits.api import (Instance, Float, push_exception_handler, Bool, Int)
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include)

from enable.api import Component, ComponentEditor
from pyface.api import ImageResource
from chaco.api import (DataRange1D, PlotAxis, VPlotContainer, create_line_plot,
                       LogMapper, ArrayPlotData, Plot, HPlotContainer)

import numpy as np

from neurogen.util import db
from neurogen import block_definitions as blocks
from neurogen.calibration.util import (psd, psd_freq, tone_power_conv_nf)
from cochlear.nidaqmx import (DAQmxDefaults, DAQmxAttenControl)
from cochlear import nidaqmx as ni
from cochlear import tone_calibration as tc


from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment)
from experiment.coroutine import coroutine, blocked, counter
from experiment.evaluate import choice, expr

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 200e3
ADC_FS = 200e3



def dp_freq(start, end, octave_spacing, probe_duration, c=1):
    frequencies = expr.octave_space(start, end, octave_spacing)
    frequencies = expr.imul(frequencies, 1.0/probe_duration)
    return choice.ascending(frequencies, c=c)


def dpoae_analyze(waveforms, fs, frequencies, mic_cal, window=None):
    results = {}
    for f in frequencies:
        nf_rms, f_rms = tone_power_conv_nf(waveforms, fs, f, window=window)
        nf_spl, f_spl = mic_cal.get_spl(f, [nf_rms.mean(), f_rms.mean()])
        results[f] = nf_spl, f_spl
    return results


@coroutine
def dpoae_reject(fs, dpoae, mic_cal, noise_floor, target):
    '''
    Accept data if DPOAE amplitude is greater than the noise floor or the noise
    floor is less than the specified value.
    '''
    while True:
        raw_data = (yield)
        data = raw_data.mean(axis=0)[0]
        nf_rms, dp_rms = tone_power_conv_nf(data, fs, dpoae)
        nf_spl, dp_spl = mic_cal.get_spl(dpoae, [nf_rms, dp_rms])
        if (nf_spl < noise_floor) or (nf_spl < dp_spl):
            target.send(raw_data)


class DPOAEData(AbstractData):

    waveform_node = Instance('tables.EArray')

    def log_trial(self, waveforms, fs, **kwargs):
        super(DPOAEData, self).log_trial(**kwargs)
        if self.waveform_node is None:
            shape = [0] + list(waveforms.shape)
            self.waveform_node = self.fh.create_earray(
                self.store_node, 'waveforms', tables.Float32Atom(), shape)
            self.waveform_node._v_attrs['fs'] = fs
        self.waveform_node.append(waveforms[np.newaxis])


class DPOAEParadigm(AbstractParadigm):

    kw = dict(context=True, log=True)

    dp_frequency = Expression('f2_frequency-f1_frequency',
                              label='DP frequency (Hz)', **kw)
    dpoae_frequency = Expression('2*f1_frequency-f2_frequency',
                                 label='DPOAE frequency (Hz)', **kw)
    f1_frequency = Expression('imul(f2_frequency/1.2, 1/response_window)',
                              label='f1 frequency (Hz)', **kw)
    f2_frequency = Expression( 'u(dp(4e3, 32e3, 0.5, response_window), f2_level)',
                              label='f2 frequency (Hz)', **kw)
    f1_level = Expression('f2_level+10', label='f1 level (dB SPL)', **kw)
    f2_level = Expression('exact_order(np.arange(0, 85, 5), c=1)',
                          label='f2 level (dB SPL)', **kw)
    dpoae_noise_floor = Expression(0, label='DPOAE noise floor (dB SPL)', **kw)

    probe_duration = Expression('response_window+response_offset*2',
                                label='Probe duration (s)', **kw)
    ramp_duration = Expression(5e-3, label='Ramp duration (s)', **kw)

    response_window = Expression(100e-3, label='Response window (s)', **kw)
    response_offset = Expression('ramp_duration*4', label='Response offset (s)',
                                 **kw)

    iti = Expression(0.01, label='Intertrial interval (s)', **kw)
    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', **kw)

    # Signal acquisition settings.  Increasing time_averages increases SNR by
    # sqrt(N).  Increasing spectral averages reduces variance of result. 8&4
    time_averages = Float(4, label='Time avg. (decr. noise floor)',
                          **kw)
    spectrum_averages = Float(4,
                              label='Spectrum avg. (decr. variability)',
                              **kw)

    traits_view = View(
        VGroup(
            VGroup(
                'time_averages',
                'spectrum_averages',
                'dpoae_noise_floor',
                'probe_duration',
                'response_offset',
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

    mic_cal = Instance('neurogen.calibration.Calibration')
    f1_sens = Instance('neurogen.calibration.PointCalibration')
    f2_sens = Instance('neurogen.calibration.PointCalibration')
    iface_adc = Instance('cochlear.nidaqmx.TriggeredDAQmxSource')
    iface_dac = Instance('cochlear.nidaqmx.DAQmxPlayer')

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)
    done = Bool(False)
    stop_requested = Bool(False)
    f2_frequency_changed = Bool(False)

    current_valid_repetitions = Int(0)
    current_repetitions = Int(0)

    extra_context = {
        'dp': dp_freq,
    }

    extra_dtypes = [
        ('measured_f1_spl', np.float32),
        ('measured_f2_spl', np.float32),
        ('measured_dp_spl', np.float32),
        ('measured_dpoae_spl', np.float32),
        ('measured_dp_nf', np.float32),
        ('measured_dpoae_nf', np.float32),
    ]

    def get_dtypes(self):
        return self.extra_dtypes

    def stop_experiment(self, info=None):
        self.stop_requested = True

    def stop(self, info=None):
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.iface_atten.clear()
        self.state = 'halted'
        self.model.data.save()

    def trial_complete(self):
        self.iface_dac.play_stop()
        self.iface_dac.clear()
        self.iface_adc.clear()
        waveforms = np.concatenate(self.waveforms, axis=0)[:self.to_acquire, 0]
        if self.f2_frequency_changed:
            self.model.clear_dp_data()
            self.f2_frequency_changed = False

        results = self.model.update_plots(
            self.adc_fs,
            waveforms,
            self.get_current_value('time_averages'),
            self.get_current_value('f1_frequency'),
            self.get_current_value('f2_frequency'),
            self.get_current_value('dpoae_frequency'),
            self.get_current_value('dp_frequency'),
            self.get_current_value('f2_level'),
            self.mic_cal,
            self.get_current_value('ramp_duration'))
        self.save_waveforms(waveforms, **results)
        try:
            self.next_trial()
        except StopIteration:
            # We are done with the experiment
            self.stop()
            return

    def save_waveforms(self, waveforms, **results):
        self.log_trial(waveforms=waveforms, fs=self.adc_fs, **results)

    def send(self, waveforms):
        self.waveforms.append(waveforms)
        self.current_valid_repetitions += len(waveforms)
        self.current_repetitions = self.pipeline.n
        if self.current_valid_repetitions >= self.to_acquire:
            raise GeneratorExit

    def poll(self):
        waveforms = self.iface_adc.read_analog()[0]
        self.dpoae_pipeline.send(waveforms)

    def set_f1_frequency(self, f1_frequency):
        # Allow the calibration to automatically handle the gain.  Since this is
        # an input gain, it must be negative.
        self.mic_cal.set_fixed_gain(-self.get_current_value('exp_mic_gain'))
        self.f1_sens = tc.tone_calibration(
            f1_frequency, self.mic_cal, gain=-20, max_thd=0.1,
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT)

    def set_f2_frequency(self, f2_frequency):
        # Allow the calibration to automatically handle the gain.  Since this is
        # an input gain, it must be negative.
        self.mic_cal.set_fixed_gain(-self.get_current_value('exp_mic_gain'))
        self.f2_sens = tc.tone_calibration(
            f2_frequency, self.mic_cal, gain=-20, max_thd=0.1,
            output_line=ni.DAQmxDefaults.SECONDARY_SPEAKER_OUTPUT)
        self.f2_frequency_changed = True

    def next_trial(self):
        self.refresh_context(evaluate=True)
        f1_frequency = self.get_current_value('f1_frequency')
        f2_frequency = self.get_current_value('f2_frequency')
        dpoae_frequency = self.get_current_value('dpoae_frequency')
        f1_level = self.get_current_value('f1_level')
        f2_level = self.get_current_value('f2_level')
        probe_duration = self.get_current_value('probe_duration')
        response_window = self.get_current_value('response_window')
        response_offset = self.get_current_value('response_offset')
        ramp_duration = self.get_current_value('ramp_duration')
        iti = self.get_current_value('iti')
        time_averages = self.get_current_value('time_averages')
        spectrum_averages = self.get_current_value('spectrum_averages')
        dpoae_nf = self.get_current_value('dpoae_noise_floor')
        mic_gain = self.get_current_value('exp_mic_gain')


        pipeline = counter(  # noqa
            blocked(time_averages, 0,
            dpoae_reject(self.adc_fs, dpoae_frequency, self.mic_cal, dpoae_nf,
            self)))

        self.waveforms = []
        self.pipeline = pipeline
        self.current_repetitions = 0
        self.current_valid_repetitions = 0
        self.to_acquire = spectrum_averages*time_averages
        self.offset_samples = int(response_offset*self.adc_fs)

        self.iface_atten = DAQmxAttenControl()

        c1 = blocks.Tone(frequency=f1_frequency, level=f1_level, name='f1') >> \
            blocks.Cos2Envelope(duration=probe_duration,
                                rise_time=ramp_duration) >> \
            ni.DAQmxChannel(calibration=self.f1_sens,
                            attenuator=self.iface_atten,
                            attenuator_channel=DAQmxDefaults.AO0_ATTEN_CHANNEL)

        c2 = blocks.Tone(frequency=f2_frequency, level=f2_level, name='f2') >> \
            blocks.Cos2Envelope(duration=probe_duration,
                                rise_time=ramp_duration) >> \
            ni.DAQmxChannel(calibration=self.f2_sens,
                            attenuator=self.iface_atten,
                            attenuator_channel=DAQmxDefaults.AO1_ATTEN_CHANNEL)

        self.iface_dac = ni.DAQmxPlayer(
            output_line=DAQmxDefaults.DUAL_SPEAKER_OUTPUT,
            duration=probe_duration+iti, fs=self.dac_fs)
        self.iface_adc = ni.TriggeredDAQmxSource(
            input_line=DAQmxDefaults.MIC_INPUT,
            fs=self.adc_fs,
            epoch_duration=response_window,
            trigger_delay=response_offset,
            pipeline=pipeline,
            complete_callback=self.trial_complete)

        # Ordering is important.  First channel is sent to ao0, second channel
        # to ao1.  The left attenuator channel controls ao0, right attenuator
        # channel controls ao1.
        self.iface_dac.add_channel(c1, name='primary')
        self.iface_dac.add_channel(c2, name='secondary')

        self.iface_atten.setup()
        self.iface_dac.set_best_attenuations()
        self.iface_atten.clear()

        try:
            self.iface_dac.queue_init('FIFO')
            self.iface_dac.queue_append(np.inf)
        except SystemError:
            self.iface_dac.clear()
            self.iface_adc.clear()
            try:
                self.next_trial()
            except StopIteration:
                # We are done with the experiment
                self.stop()
                return

        self.iface_adc.start()
        self.iface_dac.play_queue()


class DPOAEExperiment(AbstractExperiment):

    paradigm = Instance(DPOAEParadigm, ())
    data = Instance(AbstractData, ())

    time_plot = Instance(Component)
    spectrum_plot = Instance(Component)
    dp_plot = Instance(Component)
    dp_data = Instance(ArrayPlotData)

    def _dp_data_default(self):
        return ArrayPlotData(f2_level=[], f1_spl=[], f2_spl=[], dpoae_spl=[],
                             dp_spl=[], dpoae_nf=[], dp_nf=[])

    def _dp_plot_default(self):
        plot = Plot(self.dp_data)
        for pt in ('scatter', 'line'):
            plot.plot(('f2_level', 'f1_spl'), type=pt, color='red')
            plot.plot(('f2_level', 'f2_spl'), type=pt, color='crimson')
            plot.plot(('f2_level', 'dpoae_spl'), type=pt, color='black')
            plot.plot(('f2_level', 'dp_spl'), type=pt, color='darkblue')
            plot.plot(('f2_level', 'dpoae_nf'), type=pt, color='gray')
            plot.plot(('f2_level', 'dp_nf'), type=pt, color='lightblue')

        return plot

    def clear_dp_data(self, **kwargs):
        for name in self.dp_data.list_data():
            self.dp_data.set_data(name, [])

    def append_dp_data(self, **kwargs):
        for k, v in kwargs.items():
            new_data = np.append(self.dp_data.get_data(k), v)
            self.dp_data.set_data(k, new_data)

    def update_time_plots(self, fs, waveforms, f2_frequency):
        samples = waveforms.shape[-1]
        time = np.arange(samples)/fs
        waveform = waveforms.mean(axis=0)*1e3

        container = VPlotContainer(padding=50, spacing=50)
        plot = create_line_plot((time, waveform), color='black')
        plot.index_range = DataRange1D(low_setting=10e-3,
                                       high_setting=10e-3+20/f2_frequency)
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (mV)')
        plot.underlays.append(axis)
        container.add(plot)

        plot = create_line_plot((time, waveform), color='black')
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (mV)')
        plot.underlays.append(axis)
        container.add(plot)
        self.time_plot = container

    def update_spectrum_plots(self, fs, waveforms, time_averages,
                              dpoae_frequency, window):
        index_range = DataRange1D(low_setting=500, high_setting=40e3)
        index_mapper = LogMapper(range=index_range)

        samples = waveforms.shape[-1]
        w = waveforms.reshape((time_averages, -1, samples)).mean(axis=0)
        w_freq = psd_freq(w, fs)
        w_psd = psd(w, fs, window).mean(axis=0)

        container = HPlotContainer(padding=50, spacing=50)
        plot = create_line_plot((w_freq[1:], db(w_psd[1:], 1e-3)),
                                color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='Frequency (Hz)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (dB re 1mV)')
        plot.underlays.append(axis)
        container.add(plot)

        plot = create_line_plot((w_freq[1:], db(w_psd[1:], 1e-3)),
                                color='black')
        index_range = DataRange1D(low_setting=dpoae_frequency/1.1,
                                  high_setting=dpoae_frequency*1.1)
        index_mapper = LogMapper(range=index_range)
        plot.index_mapper = index_mapper
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='Frequency (Hz)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (dB re 1mV)')
        plot.underlays.append(axis)
        container.add(plot)

        self.spectrum_plot = container

    def update_plots(self, fs, waveforms, time_averages, f1_frequency,
                     f2_frequency, dpoae_frequency, dp_frequency, f2_level,
                     mic_cal, ramp_duration, window=None):

        self.update_time_plots(fs, waveforms, f2_frequency)
        self.update_spectrum_plots(fs, waveforms, time_averages,
                                   dpoae_frequency, window)
        frequencies = f1_frequency, f2_frequency, dpoae_frequency, dp_frequency
        samples = waveforms.shape[-1]
        w = waveforms.reshape((time_averages, -1, samples)).mean(axis=0)
        results = dpoae_analyze(w, fs, frequencies, mic_cal, window=window)
        self.append_dp_data(f1_spl=results[f1_frequency][1],
                            f2_spl=results[f2_frequency][1],
                            dpoae_nf=results[dpoae_frequency][0],
                            dpoae_spl=results[dpoae_frequency][1],
                            dp_nf=results[dp_frequency][0],
                            dp_spl=results[dp_frequency][1],
                            )
        self.append_dp_data(f2_level=f2_level)
        return {
            'measured_f1_spl': results[f1_frequency][1],
            'measured_f2_spl': results[f2_frequency][1],
            'measured_dp_spl': results[dp_frequency][1],
            'measured_dpoae_spl': results[dpoae_frequency][1],
            'measured_dp_nf': results[dp_frequency][0],
            'measured_dpoae_nf': results[dpoae_frequency][0],
        }

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
                    Item('handler.current_valid_repetitions', style='readonly',
                         label='Valid repetitions')
                ),
                show_border=True,
            ),
            VGroup(
                HGroup(
                    Item('time_plot', show_label=False,
                         editor=ComponentEditor(width=300, height=300)),
                    Item('dp_plot', show_label=False,
                         editor=ComponentEditor(width=300, height=300)),
                ),
                Item('spectrum_plot', show_label=False,
                     editor=ComponentEditor(width=300, height=300)),
            )
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


def launch_gui(mic_cal, filename, **kwargs):
    with tables.open_file(filename, 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        paradigm = DPOAEParadigm()
        experiment = DPOAEExperiment(data=data, paradigm=paradigm)
        controller = DPOAEController(mic_cal=mic_cal)
        experiment.edit_traits(handler=controller, **kwargs)


def configure_logging(filename):
    time_format = '[%(asctime)s] :: %(name)s - %(levelname)s - %(message)s'
    simple_format = '%(name)s - %(message)s'

    logging_config = {
        'version': 1,
        'formatters': {
            'time': {'format': time_format},
            'simple': {'format': simple_format},
            },
        'handlers': {
            # This is what gets printed out to the console
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'DEBUG',
                },
            # This is what gets saved to the file
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'time',
                'filename': filename,
                'level': 'DEBUG',
                }
            },
        'loggers': {
            'cochlear.tone_calibration': {'level': 'DEBUG'},
            'tone_calibration': {'level': 'DEBUG'},
            },
        'root': {
            'handlers': ['console', 'file'],
            },
        }
    logging.config.dictConfig(logging_config)


if __name__ == '__main__':
    import logging.config
    configure_logging('temp.log')
    import PyDAQmx as pyni
    pyni.DAQmxResetDevice('Dev1')
    from neurogen.calibration import InterpCalibration
    c = InterpCalibration.from_mic_file('c:/data/cochlear/calibration/141112 DPOAE frequency calibration in half-octaves 500 to 32000.mic')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        experiment = DPOAEExperiment(paradigm=DPOAEParadigm(), data=data)
        controller = DPOAEController(mic_cal=c)
        experiment.configure_traits(handler=controller)
