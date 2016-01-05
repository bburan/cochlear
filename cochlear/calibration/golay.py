from traits.api import Int, Float
from traitsui.api import VGroup, Item
from enable.api import ComponentEditor

from scipy import signal
import numpy as np

from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear.calibration import GolayCalibration
from cochlear.calibration.base import (BaseSignalSettings,
                                       BaseSignalController,
                                       BaseSignalExperiment,
                                       HRTFControllerMixin,
                                       HRTFSettingsMixin,
                                       HRTFExperimentMixin,
                                       ReferenceControllerMixin,
                                       ReferenceSettingsMixin)


################################################################################
# Base classes supporting both Golay cal tests
################################################################################
class BaseGolaySettings(BaseSignalSettings):

    kw = dict(context=True)

    n = Int(14, label='Bits', **kw)
    fft_averages = 4
    waveform_averages = 2
    iti = 1e-5
    ab_delay = Float(2, label='Delay between code A and B (sec)', **kw)
    smoothing_window = Int(25, label='Frequency smoothing window size', **kw)

    stimulus_settings = VGroup(
        'n',
        label='Golay settings',
        show_border=True
    )

    presentation_settings = VGroup(
        'discard',
        'fft_averages',
        'waveform_averages',
        'iti',
        'ab_delay',
        'smoothing_window',
        label='Presentation settings',
        show_border=True,
    )


class BaseGolayController(BaseSignalController):

    def setup_acquire(self):
        # Load the variables
        output = self.get_current_value('output')
        analog_output = '/{}/{}'.format(ni.DAQmxDefaults.DEV, output)
        self.iface = GolayCalibration(
            ab_delay=self.get_current_value('ab_delay'),
            n=self.get_current_value('n'),
            vrms=self.get_current_value('amplitude'),
            gain=self.get_current_value('output_gain'),
            repetitions=self.get_current_value('averages'),
            iti=self.get_current_value('iti'),
            fs=self.adc_fs,
            output_line=analog_output,
            input_line=self.MIC_INPUT,
            callback=self.poll,
        )


################################################################################
# Reference microphone calibration
################################################################################
class ReferenceCalibrationSettings(ReferenceSettingsMixin, BaseGolaySettings):

    output_gain = 6
    discard = 2
    n = 13


class ReferenceCalibrationController(ReferenceControllerMixin,
                                     BaseGolayController):

    def finalize(self):
        discard = self.get_current_value('discard')
        smoothing_window = self.get_current_value('smoothing_window')
        ref_mic_sens = self.get_current_value('ref_mic_sens')
        ref_mic_gain = dbi(self.get_current_value('ref_mic_gain'))
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=[exp_mic_gain, ref_mic_gain],
                                     discard=discard)

        exp_mic_waveform, ref_mic_waveform = \
            results['mic_waveforms'].mean(axis=0)

        exp_mic_psd, ref_mic_psd = db(results['tf_psd'])
        exp_mic_phase, ref_mic_phase = results['tf_phase']
        exp_mic_sens = exp_mic_psd+db(ref_mic_sens)-ref_mic_psd
        if smoothing_window > 0:
            w = signal.hanning(smoothing_window)
            w /= w.sum()
            exp_mic_sens = np.convolve(exp_mic_sens, w, mode='same')

        results['exp_mic_waveform'] = exp_mic_waveform
        results['ref_mic_waveform'] = ref_mic_waveform
        results['ref_mic_psd'] = ref_mic_psd
        results['exp_mic_psd'] = exp_mic_psd
        results['ref_mic_phase'] = ref_mic_phase
        results['exp_mic_phase'] = exp_mic_phase
        results['exp_mic_sens'] = exp_mic_sens
        results['speaker_spl'] = ref_mic_psd-db(ref_mic_sens)-db(20e-6)
        results['frequency'] = results['mic_frequency']

        self.model.update_plots(results)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class ReferenceCalibration(BaseSignalExperiment):
    pass


def reference_calibration(**kwargs):
    controller = ReferenceCalibrationController()
    paradigm = ReferenceCalibrationSettings(output='ao0')
    ReferenceCalibration(paradigm=paradigm) \
        .configure_traits(handler=controller, **kwargs)


################################################################################
# Reference microphone calibration
################################################################################
class HRTFSettings(HRTFSettingsMixin, BaseGolaySettings):

    n = 12
    discard = 1
    ab_delay = 0.1
    exp_mic_gain = 40
    output_gain = -20
    fft_averages = 2


class HRTFController(HRTFControllerMixin, BaseGolayController):

    def finalize(self):
        discard = self.get_current_value('discard')
        smoothing_window = self.get_current_value('smoothing_window')
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=exp_mic_gain, discard=discard)

        exp_mic_waveform = results['mic_waveforms'].mean(axis=0)[0]
        exp_mic_psd = db(results['tf'])[0]
        if smoothing_window > 0:
            w = signal.hanning(smoothing_window)
            w /= w.sum()
            exp_mic_psd = np.convolve(exp_mic_psd, w, mode='same')

        speaker_spl = self.calibration.get_spl(results['mic_frequency'],
                                               results['tf'][0])

        results['exp_mic_waveform'] = exp_mic_waveform
        results['exp_mic_psd'] = exp_mic_psd
        results['frequency'] = results['mic_frequency']
        results['speaker_spl'] = speaker_spl

        self.model.update_plots(results, freq_lb=500, freq_ub=50e3)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class HRTF(HRTFExperimentMixin, BaseSignalExperiment):

    def _sig_waveform_plot_default(self):
        plot = super(HRTF, self)._sig_waveform_plot_default()
        plot.index_range.high_setting = 1e-3
        return plot

    def _mic_waveform_plot_default(self):
        plot = super(HRTF, self)._mic_waveform_plot_default()
        plot.index_range.high_setting = 1e-3
        return plot

    signal_plots = VGroup(
        Item('sig_waveform_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        Item('speaker_spl_plot', editor=ComponentEditor(),
                width=500, height=200, show_label=False),
        label='Signal',
    )


def hrtf_calibration(calibration, filename, **kwargs):
    controller = HRTFController(calibration=calibration, filename=filename)
    paradigm = HRTFSettings()
    HRTF(paradigm=paradigm).configure_traits(handler=controller, **kwargs)


if __name__ == '__main__':
    reference_calibration()
    #import os.path
    #from neurogen.calibration import InterpCalibration
    #mic_file = os.path.join('c:/data/cochlear/calibration',
    #                        '150730 - Golay calibration with 377C10.mic')
    #c = InterpCalibration.from_mic_file(mic_file)
    #hrtf_calibration(c, 'temp.hdf5')
