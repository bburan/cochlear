import logging
log = logging.getLogger(__name__)

import time

from traits.api import (Instance, Float, push_exception_handler, Bool, Int,
                        List, HasTraits, Str, Property)
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu, Tabbed, HGroup, Include,
                          ListEditor)

from enable.api import Component, ComponentEditor
from pyface.api import ImageResource, error
from chaco.api import (DataRange1D, PlotAxis, PlotGrid, VPlotContainer,
                       create_line_plot, LogMapper, ArrayPlotData, Plot,
                       HPlotContainer)

import numpy as np
from scipy import signal

from functools import partial

from neurogen.util import db
from neurogen import block_definitions as blocks
from neurogen.calibration.util import (psd, psd_freq, tone_power_conv_nf)
from cochlear import nidaqmx as ni
from cochlear import calibration

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, depends_on)
from experiment.coroutine import (coroutine, blocked, counter, broadcast, call,
                                  create_pipeline, accumulate)
from experiment.evaluate import choice, expr

import tables

from pkg_resources import resource_filename
icon_dir = [resource_filename('experiment', 'icons')]

push_exception_handler(reraise_exceptions=True)

DAC_FS = 100e3
ADC_FS = 100e3

def dp_freq(start, end, octave_spacing, period, c=1):
    frequencies = expr.octave_space(start, end, octave_spacing)
    frequencies = expr.imul(frequencies, period)
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
        log.debug('DPOAE reject: DPOAE %.2f, noise floor %.2f', dp_spl, nf_spl)
        if (nf_spl < noise_floor) or (nf_spl < dp_spl):
            target.send(raw_data)


@coroutine
def overlap(fraction, target):
    data = (yield)
    shape = data.shape
    samples = shape[-1]
    skip_samples = int(samples*fraction)
    while True:
        while data.shape[-1] >= samples:
            target.send(data[..., :samples])
            data = data[..., skip_samples:]
        new_data = (yield)
        data = np.concatenate((data, new_data), axis=-1)


@coroutine
def discard_first(target):
    # Nothing happens because we're discarding the first one ...
    data = (yield)
    while True:
        data = (yield)
        target.send(data)


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

    f1_frequency = Expression('imul(f2_frequency/1.2, 1/response_window)',
                              label='f1 frequency (Hz)', **kw)
    f2_frequency = Expression(
        'u(dp(0.7e3, 16e3, 0.5, 1/response_window), f2_level)',
        label='f2 frequency (Hz)', **kw)

    f1_level = Expression('f2_level+10', label='f1 level (dB SPL)', **kw)
    f2_level = Expression('exact_order(np.arange(10, 85, 5), c=1)',
                          label='f2 level (dB SPL)', **kw)
    dpoae_noise_floor = Expression(0, label='DPOAE noise floor (dB SPL)', **kw)
    response_window = Expression('50e-3', label='Response window (s)', **kw)
    exp_mic_gain = Float(40, label='Exp. mic. gain (dB)', **kw)

    # Signal acquisition settings.  Increasing time_averages increases SNR by
    # sqrt(N).  Increasing spectral averages reduces variance of result.  EPL
    # uses 8&4.
    time_averages = Float(8, label='Time avg. (decr. noise floor)', **kw)
    spectrum_averages = Float(4, label='Spectrum avg. (decr. variability)',
                              **kw)

    traits_view = View(
        VGroup(
            VGroup(
                'time_averages',
                'spectrum_averages',
                'dpoae_noise_floor',
                'response_window',
                'exp_mic_gain',
                label='Acquisition settings',
                show_border=True,
            ),
            VGroup(
                'f1_frequency',
                'f2_frequency',
                'f1_level',
                'f2_level',
                label='Stimulus settings',
                show_border=True,
            ),
        ),
    )


class DPOAEController(AbstractController):

    mic_cal = Instance('neurogen.calibration.Calibration')
    primary_sens = Instance('neurogen.calibration.PointCalibration')
    secondary_sens = Instance('neurogen.calibration.PointCalibration')
    iface_adc = Instance('cochlear.nidaqmx.DAQmxInput')
    iface_dac = Instance('cochlear.nidaqmx.ContinuousDAQmxPlayer')

    kw = dict(log=True, dtype=np.float32)
    primary_spl = Float(label='Primary @ 1Vrms, 0dB att (dB SPL)', **kw)
    secondary_spl = Float(label='Secondary @ 1Vrms, 0dB att (dB SPL)', **kw)
    primary_attenuation = Float(label='Primary attenuation (dB)', **kw)
    secondary_attenuation = Float(label='Secondary attenuation (dB)', **kw)
    start_time = Float(label='Start time', **kw)
    end_time = Float(label='End time', **kw)
    dpoae_frequency = Float(0, label='DPOAE frequency (Hz)', **kw)
    dp_frequency = Float(0, label='DP frequency (Hz)', **kw)

    current_valid_repetitions = Int(0, log=True, dtype=np.int32)
    current_repetitions = Int(0, log=True, dtype=np.int32)

    adc_fs = Float(ADC_FS)
    dac_fs = Float(DAC_FS)
    done = Bool(False)

    f2_frequency_changed = Bool(False)

    # Provide the function, `dp_freq`, to the experiment namespace under the
    # alias `dp`.
    extra_context = {
        'dp': dp_freq,
    }

    # Extra columns to add to the trial log.
    extra_dtypes = [
        ('measured_f1_spl', np.float32),
        ('measured_f2_spl', np.float32),
        ('measured_dp_spl', np.float32),
        ('measured_dpoae_spl', np.float32),
        ('measured_dp_nf', np.float32),
        ('measured_dpoae_nf', np.float32),
        ('primary_sens', np.float32),
        ('secondary_sens', np.float32),
    ]

    primary_calibration_gain = Property(log=True, dtype=np.float32)
    secondary_calibration_gain = Property(log=True, dtype=np.float32)

    def _get_primary_calibration_gain(self):
        frequency = self.get_current_value('f2_frequency')
        #return -20 if frequency <= 700 else -40
        return -40

    def _get_secondary_calibration_gain(self):
        frequency = self.get_current_value('f1_frequency')
        #return -20 if frequency <= 700 else -40
        return -40

    def next_trial(self, info=None):
        try:
            log.debug('Preparing next trial')
            self.refresh_context(evaluate=True)
        except StopIteration:
            self.stop()
            return
        except Exception as e:
            raise
            error(None, str(e))
            self.stop()
            return

        response_window = self.get_current_value('response_window')
        f1_frequency = self.get_current_value('f1_frequency')
        f2_frequency = self.get_current_value('f2_frequency')
        if self.value_changed('f1_frequency') or \
                self.value_changed('f2_frequency'):
            self.calibrate_speakers(f1_frequency, f2_frequency)

        self.dpoae_frequency = 2*f1_frequency-f2_frequency
        self.dp_frequency = f2_frequency-f1_frequency

        f1_level = self.get_current_value('f1_level')
        f2_level = self.get_current_value('f2_level')
        time_averages = self.get_current_value('time_averages')
        spectrum_averages = self.get_current_value('spectrum_averages')
        dpoae_nf = self.get_current_value('dpoae_noise_floor')
        mic_gain = self.get_current_value('exp_mic_gain')

        analysis_samples = int(self.adc_fs*response_window)

        # Pipeline continously recieves data from the analog acquisition system,
        # groups the samples into chunks of `analysis_samples`.  Each chunk is
        # then sent through the rest of the pipeline.  The counter tracks the
        # number of chunks that have pased through, reshape ensures that the
        # next blocking can group the chunks into the specified number of
        # `time_averages` (for analysis by the DPOAE reject function).  If the
        # chunks pass the reject criterion, then they are returned to this class
        # for storage.
        accumulate_epoch = partial(blocked, analysis_samples, -1)
        accumulate_time = partial(blocked, time_averages, 0)
        screen_epoch = partial(dpoae_reject, self.adc_fs, self.dpoae_frequency,
                               self.mic_cal, dpoae_nf)
        save_epoch = call(self.epoch_acquisition)
        count_epochs = counter(self.update_repetitions)

        pipeline = create_pipeline(
            accumulate_epoch,
            discard_first,
            broadcast(
                count_epochs,
                create_pipeline(
                    accumulate_time,
                    screen_epoch,
                    save_epoch
                )
            )
        )

        self.waveforms = []
        self.pipeline = pipeline
        self.current_repetitions = 0
        self.current_valid_repetitions = 0
        self.to_acquire = spectrum_averages*time_averages

        attenuator = ni.DAQmxBaseAttenuator()

        c1 = blocks.Tone(frequency=f1_frequency, level=f1_level, name='f1') >> \
            blocks.Cos2Envelope(duration=np.inf, rise_time=10e-3) >> \
            ni.DAQmxChannel(calibration=self.primary_sens,
                            attenuator=attenuator, attenuator_channel=0)

        c2 = blocks.Tone(frequency=f2_frequency, level=f2_level, name='f2') >> \
            blocks.Cos2Envelope(duration=np.inf, rise_time=10e-3) >> \
            ni.DAQmxChannel(calibration=self.secondary_sens,
                            attenuator=attenuator, attenuator_channel=1)

        self.iface_dac = ni.ContinuousDAQmxPlayer(
            output_line=ni.DAQmxDefaults.DUAL_SPEAKER_OUTPUT,
            fs=self.dac_fs,
            # TODO: Hack -- should be able to set this to infinity (e.g. never
            # halt).  However, the time() vector is sometimes used in
            # ContinuousPlayer.
            duration=120,
            buffer_size=10,
            monitor_interval=1,
            expected_range=10,
        )

        self.iface_adc = ni.DAQmxInput(
            fs=self.adc_fs,
            input_line=ni.DAQmxDefaults.MIC_INPUT,
            callback_samples=analysis_samples,
            pipeline=pipeline,
            expected_range=10,
            done_callback=self.trial_complete,
            start_trigger='ao/StartTrigger',
        )

        # Ordering is important.  First channel is sent to ao0, second channel
        # to ao1.  The left attenuator channel controls ao0, right attenuator
        # channel controls ao1.
        self.iface_dac.add_channel(c1, name='primary')
        self.iface_dac.add_channel(c2, name='secondary')

        log.debug('Setting hardware attenuation')
        self.primary_attenuation, self.secondary_attenuation = \
            self.iface_dac.set_best_attenuations()

        self.start_time = time.time()
        log.debug('Starting ADC acquisition')
        self.iface_adc.start()
        log.debug('Starting DAC playout')
        self.iface_dac.play_continuous()

    def update_repetitions(self, repetitions):
        self.current_repetitions = repetitions

    def stop_experiment(self, info=None):
        if self.iface_dac is not None:
            self.iface_dac.clear()
            self.iface_adc.clear()
            self.model.data.save()

    def trial_complete(self):
        self.end_time = time.time()
        self.iface_dac.stop()
        self.iface_dac.clear()
        self.iface_adc.clear()
        if self.f2_frequency_changed:
            self.model.clear_dp_data(str(self.get_current_value('f2_frequency')))
            self.f2_frequency_changed = False
        self.f1_frequency_changed = False

        waveforms = np.concatenate(self.waveforms, axis=0)
        waveforms = waveforms[:self.to_acquire, 0]
        results = self.model.update_plots(
            self.adc_fs,
            waveforms,
            self.get_current_value('time_averages'),
            self.get_current_value('f1_frequency'),
            self.get_current_value('f2_frequency'),
            self.dpoae_frequency,
            self.dp_frequency,
            self.get_current_value('f2_level'),
            self.mic_cal
        )
        self.save_waveforms(waveforms, **results)

        if not self.pause_requested:
            self.next_trial()
        else:
            self.state = 'paused'

    def next_parameter(self, info=None):
        self.iface_dac.stop()
        self.iface_dac.clear()
        self.iface_adc.clear()
        if not self.pause_requested:
            self.next_trial()
        else:
            self.state = 'paused'

    def save_waveforms(self, waveforms, **results):
        primary_sens = self.primary_sens.get_sens(
            self.get_current_value('f1_frequency'))
        secondary_sens = self.secondary_sens.get_sens(
            self.get_current_value('f2_frequency'))
        self.log_trial(
            waveforms=waveforms,
            fs=self.adc_fs,
            primary_sens=primary_sens,
            secondary_sens=secondary_sens,
            primary_spl=self.primary_spl,
            secondary_spl=self.secondary_spl,
            primary_calibration_gain=self.primary_calibration_gain,
            secondary_calibration_gain=self.secondary_calibration_gain,
            #primary_attenuation=self.primary_attenuation,
            #secondary_attenuation=self.secondary_attenuation,
            total_repetitions=self.current_repetitions,
            **results)

    def epoch_acquisition(self, waveforms):
        log.debug('Recieved %r samples', waveforms.shape)
        self.waveforms.append(waveforms)
        self.current_valid_repetitions += len(waveforms)
        if self.current_valid_repetitions >= self.to_acquire:
            raise GeneratorExit

    def set_exp_mic_gain(self, exp_mic_gain):
        # Allow the calibration to automatically handle the gain.  Since this is
        # an input gain, it must be negative (meaning that the actual measured
        # value is less).
        self.mic_cal.set_fixed_gain(-exp_mic_gain)

    def set_f2_frequency(self, f2_frequency):
        self.f2_frequency_changed = True

    @depends_on('exp_mic_gain')
    def calibrate_speakers(self, f1_frequency, f2_frequency):
        log.debug('Calibrating speakers')
        f1_sens, f2_sens = calibration.two_tone_calibration(
            f1_frequency, f2_frequency, self.mic_cal,
            f1_gain=self.primary_calibration_gain,
            f2_gain=self.secondary_calibration_gain,
            max_thd=None)
        self.primary_sens = f1_sens
        self.secondary_sens = f2_sens
        self.primary_spl = self.primary_sens.get_spl(f1_frequency, 1)
        self.secondary_spl = self.secondary_sens.get_spl(f2_frequency, 1)


class _DPPlot(HasTraits):

    parameter = Str
    plot = Instance(Component)

    traits_view = View(
        Item('plot', show_label=False,
             editor=ComponentEditor(width=300, height=300))
    )


class DPOAEExperiment(AbstractExperiment):

    paradigm = Instance(DPOAEParadigm, ())
    data = Instance(AbstractData, ())

    tf_data = Instance(ArrayPlotData)

    time_plots = Instance(Component)
    zoomed_time_plot = Instance(Component)
    overview_time_plot = Instance(Component)

    spectrum_plots = Instance(Component)
    dpoae_spectrum_plot = Instance(Component)
    mic_spectrum_plot= Instance(Component)

    dp_plots = List(Instance(_DPPlot))
    dp_data = Instance(ArrayPlotData)

    def _tf_data_default(self):
        return ArrayPlotData(time=[], first_waveform=[], last_waveform=[],
                             average_waveform=[], frequency=[], spectrum=[])

    def _dp_data_default(self):
        return ArrayPlotData(f2_level=[], f1_spl=[], f2_spl=[], dpoae_spl=[],
                             dp_spl=[], dpoae_nf=[], dp_nf=[])

    def _dp_plot_default(self, parameter):
        plot = Plot(self.dp_data)
        for pt in ('scatter', 'line'):
            plot.plot(('f2_level', 'f1_spl'), type=pt, color='red')
            plot.plot(('f2_level', 'f2_spl'), type=pt, color='crimson')
            plot.plot(('f2_level', 'dpoae_spl'), type=pt, color='black')
            plot.plot(('f2_level', 'dp_spl'), type=pt, color='darkblue')
            plot.plot(('f2_level', 'dpoae_nf'), type=pt, color='gray')
            plot.plot(('f2_level', 'dp_nf'), type=pt, color='lightblue')
        plot.underlays = []
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='f2 level (dB SPL)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='DPOAE level (dB SPL)')
        plot.underlays.append(axis)
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                        orientation='vertical', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                        orientation='horizontal', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        return _DPPlot(plot=plot, parameter=parameter)

    def _zoomed_time_plot_default(self):
        plot = Plot(self.tf_data, padding=0, spacing=0)
        plot.plot(('time', 'average_waveform'), type='line', color='black')
        plot.plot(('time', 'first_waveform'), type='line', color='blue')
        plot.plot(('time', 'last_waveform'), type='line', color='red')
        plot.underlays = []
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (mV)')
        plot.underlays.append(axis)
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                        orientation='vertical', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                        orientation='horizontal', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        return plot

    def _overview_time_plot_default(self):
        plot = Plot(self.tf_data, padding=0, spacing=0)
        plot.plot(('time', 'average_waveform'), type='line', color='black')
        plot.underlays = []
        axis = PlotAxis(orientation='bottom', component=plot,
                        tick_label_formatter=lambda x: "{:.2f}".format(x*1e3),
                        title='Time (msec)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (mV)')
        plot.underlays.append(axis)
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                        orientation='vertical', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                        orientation='horizontal', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        return plot

    def _time_plots_default(self):
        container = VPlotContainer(padding=50, spacing=50,
                                   bgcolor='transparent')
        container.add(self.zoomed_time_plot)
        container.add(self.overview_time_plot)
        return container

    def _mic_spectrum_plot_default(self):
        plot = Plot(self.tf_data)
        plot.plot(('frequency', 'spectrum'), type='line', color='black',
                  index_scale='log')
        plot.index_range = DataRange1D(low_setting=500, high_setting=50e3)
        plot.underlays = []
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='Frequency (Hz)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (dB re 1mV)')
        plot.underlays.append(axis)
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                        orientation='vertical', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                        orientation='horizontal', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        return plot

    def _dpoae_spectrum_plot_default(self):
        plot = Plot(self.tf_data)
        plot.plot(('frequency', 'spectrum'), type='line', color='black',
                  index_scale='log')
        plot.underlays = []
        axis = PlotAxis(orientation='bottom', component=plot,
                        title='Frequency (Hz)')
        plot.underlays.append(axis)
        axis = PlotAxis(orientation='left', component=plot,
                        title='Exp. mic. (dB re 1mV)')
        plot.underlays.append(axis)
        grid = PlotGrid(mapper=plot.index_mapper, component=plot,
                        orientation='vertical', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        grid = PlotGrid(mapper=plot.value_mapper, component=plot,
                        orientation='horizontal', line_style='dot',
                        line_color='lightgray')
        plot.underlays.append(grid)
        return plot

    def _spectrum_plots_default(self):
        container = HPlotContainer(padding=10, spacing=10,
                                   bgcolor='transparent')
        container.add(self.mic_spectrum_plot)
        container.add(self.dpoae_spectrum_plot)
        return container

    def clear_dp_data(self, parameter):
        self.dp_data = self._dp_data_default()
        self.dp_plots.append(self._dp_plot_default(parameter))

    def append_dp_data(self, **kwargs):
        for k, v in kwargs.items():
            new_data = np.append(self.dp_data.get_data(k), v)
            self.dp_data.set_data(k, new_data)

    def update_time_plots(self, fs, waveforms, f2_frequency):
        waveforms = signal.detrend(waveforms)*1e3
        samples = waveforms.shape[-1]
        time = np.arange(samples)/fs
        self.tf_data.set_data('time', time)
        self.tf_data.set_data('average_waveform', waveforms.mean(axis=0))
        self.tf_data.set_data('first_waveform', waveforms[0])
        self.tf_data.set_data('last_waveform', waveforms[-1])
        self.zoomed_time_plot.index_range = DataRange1D(
            low_setting=0, high_setting=10/f2_frequency)

    def update_spectrum_plots(self, fs, waveforms, time_averages,
                              dpoae_frequency, window):
        samples = waveforms.shape[-1]
        w = waveforms.reshape((time_averages, -1, samples)).mean(axis=0)
        w_freq = psd_freq(w, fs)
        w_psd = db(psd(w, fs, window).mean(axis=0), 1e-3)
        self.tf_data.set_data('frequency', w_freq[1:])
        self.tf_data.set_data('spectrum', w_psd[1:])
        self.dpoae_spectrum_plot.index_range = DataRange1D(
            low_setting=dpoae_frequency/1.1, high_setting=dpoae_frequency*1.1)

    def update_plots(self, fs, waveforms, time_averages, f1_frequency,
                     f2_frequency, dpoae_frequency, dp_frequency, f2_level,
                     mic_cal, window=None):

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
                ),
                HGroup(
                    VGroup(
                        Item('handler.current_repetitions', style='readonly',
                            label='Repetitions'),
                        Item('handler.current_valid_repetitions',
                             style='readonly', label='Valid repetitions'),
                        show_border=True,
                    ),
                    VGroup(
                        Item('handler.primary_spl', style='readonly',
                             format_str='%0.2f'),
                        Item('handler.secondary_spl', style='readonly',
                             format_str='%0.2f'),
                        Item('handler.primary_attenuation', style='readonly',
                             format_str='%0.2f'),
                        Item('handler.secondary_attenuation', style='readonly',
                             format_str='%0.2f'),
                        show_border=True,
                        label='Diagnostics',
                    ),
                ),
            ),
            VGroup(
                HGroup(
                    Item('time_plots', show_label=False,
                         editor=ComponentEditor(width=300, height=300)),
                    Item('dp_plots', show_label=False, style='custom',
                         editor=ListEditor(use_notebook=True, deletable=False,
                                           page_name='.parameter')),
                ),
                Item('spectrum_plots', show_label=False,
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
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
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
            Action(name='Skip', action='next_parameter',
                   image=ImageResource('media_skip_forward', icon_dir),
                   enabled_when='handler.state=="running"'),
        ),
        menubar=MenuBar(
            Menu(
                ActionGroup(
                    Action(name='Load settings', action='load_paradigm'),
                    Action(name='Save settings', action='save_paradigm'),
                ),
                name='&Settings',
            ),
        ),
        id='lbhb.DPOAEExperiment',
    )


def launch_gui(mic_cal, filename, paradigm_dict=None, **kwargs):
    if filename is None:
        filename = 'dummy'
        tbkw = {'driver': 'H5FD_CORE', 'driver_core_backing_store': 0}
    else:
        tbkw = {}
    with tables.open_file(filename, 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        if paradigm_dict is None:
            paradigm_dict = {}
        paradigm = DPOAEParadigm(**paradigm_dict)
        experiment = DPOAEExperiment(data=data, paradigm=paradigm)
        controller = DPOAEController(mic_cal=mic_cal)
        experiment.edit_traits(handler=controller, **kwargs)


if __name__ == '__main__':
    import os.path
    from cochlear import configure_logging
    from neurogen.calibration import InterpCalibration, FlatCalibration
    from neurogen.util import db
    import PyDAQmx as pyni
    configure_logging('temp.log')

    pyni.DAQmxResetDevice('PXI4461')
    mic_file = os.path.join('c:/data/cochlear/calibration',
                            '150807 - Golay calibration with 377C10.mic')
    c = InterpCalibration.from_mic_file(mic_file)
    mic_input = ni.DAQmxDefaults.MIC_INPUT

    log.debug('====================== MAIN =======================')
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = DPOAEData(store_node=fh.root)
        paradigm = DPOAEParadigm(f2_frequency=2e3, time_averages=32,
                                 spectrum_averages=1)
        experiment = DPOAEExperiment(paradigm=paradigm, data=data)
        controller = DPOAEController(mic_cal=c, mic_input_line=mic_input)
        experiment.configure_traits(handler=controller)
