from traits.api import Int, Float
from traitsui.api import VGroup

from scipy import signal
import numpy as np

from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear.calibration import GolayCalibration
from cochlear.calibration.base import (BaseSignalSettings,
                                       BaseSignalController,
                                       RefMicSettingsMixin,
                                       BaseSignalExperiment)


################################################################################
# Base classes supporting both Golay cal tests
################################################################################
class BaseGolaySettings(BaseSignalSettings):

    kw = dict(context=True)

    n = Int(14, label='Bits', **kw)
    fft_averages = 8
    waveform_averages = 1
    discard = 2
    iti = 1e-5
    ab_delay = Float(2, label='Delay between code A and B (sec)', **kw)
    smoothing_window = Int(25, label='Frequency smoothing window size', **kw)

    output_gain = 12
    amplitude = 5

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

    MIC_INPUT = '{}, {}'.format(ni.DAQmxDefaults.MIC_INPUT,
                                ni.DAQmxDefaults.REF_MIC_INPUT)

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

        exp_mic_psd, ref_mic_psd = db(results['tf'])
        exp_mic_sens = exp_mic_psd+db(ref_mic_sens)-ref_mic_psd
        if smoothing_window > 0:
            w = signal.hanning(smoothing_window)
            w /= w.sum()
            exp_mic_sens = np.convolve(exp_mic_sens, w, mode='same')

        results['exp_mic_waveform'] = exp_mic_waveform
        results['ref_mic_waveform'] = ref_mic_waveform
        results['ref_mic_psd'] = ref_mic_psd
        results['exp_mic_psd'] = exp_mic_psd
        results['exp_mic_sens'] = exp_mic_sens
        results['speaker_spl'] = ref_mic_psd-db(ref_mic_sens)-db(20e-6)
        results['frequency'] = results['mic_frequency']

        self.model.update_plots(results)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class ReferenceGolaySettings(RefMicSettingsMixin, BaseGolaySettings):
    pass


class BaseGolay(BaseSignalExperiment):
    pass


def golay_calibration(**kwargs):
    controller = BaseGolayController()
    paradigm = ReferenceGolaySettings(output='ao0')
    BaseGolay(paradigm=paradigm) \
        .configure_traits(handler=controller, **kwargs)


if __name__ == '__main__':
    golay_calibration()
