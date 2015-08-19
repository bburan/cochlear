from traits.api import Float, Property
from traitsui.api import VGroup, Item

from neurogen.util import db, dbi

from cochlear import nidaqmx as ni
from cochlear import settings
from cochlear.calibration import ChirpCalibration
from cochlear.calibration.base import (BaseSignalSettings,
                                       BaseSignalController,
                                       BaseSignalExperiment,
                                       HRTFControllerMixin,
                                       HRTFSettingsMixin,
                                       HRTFExperimentMixin,
                                       ReferenceControllerMixin,
                                       ReferenceSettingsMixin)


################################################################################
# Base classes supporting both chirp cal tests
################################################################################
class BaseChirpSettings(BaseSignalSettings):

    kw = dict(context=True)

    rise_time = Float(0, label='Envelope rise time', **kw)
    freq_lb = Float(0.05e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(50e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    start_attenuation = Float(0, label='Start attenuation (dB)', **kw)
    end_attenuation = Float(0, label='End attenuation (dB)', **kw)

    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, iti',
                              label='Total duration (sec)')

    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.duration+self.iti)*self.averages

    stimulus_settings = VGroup(
        'rise_time',
        'freq_lb',
        'freq_ub',
        'freq_resolution',
        Item('duration', style='readonly'),
        Item('total_duration', style='readonly'),
        'start_attenuation',
        'end_attenuation',
        show_border=True,
        label='Chirp settings',
    )


class BaseChirpController(BaseSignalController):

    def setup_acquire(self):
        # Load the variables
        output = self.get_current_value('output')
        analog_output = '/{}/{}'.format(ni.DAQmxDefaults.DEV, output)
        self.iface = ChirpCalibration(
            freq_lb=self.get_current_value('freq_lb'),
            freq_ub=self.get_current_value('freq_ub'),
            start_atten=self.get_current_value('start_attenuation'),
            end_atten=self.get_current_value('end_attenuation'),
            vrms=self.get_current_value('amplitude'),
            gain=self.get_current_value('output_gain'),
            repetitions=self.get_current_value('averages'),
            duration=self.get_current_value('duration'),
            rise_time=self.get_current_value('rise_time'),
            iti=self.get_current_value('iti'),
            fs=self.adc_fs,
            output_line=analog_output,
            input_line=self.MIC_INPUT,
            callback=self.poll,
        )

    def save_settings(self, info=None):
        self.save_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')

    def load_settings(self, info=None):
        self.load_paradigm(settings.SETTINGS_DIR,
                           'Tone calibration settings (*.tc_par)|*.tc_par')


################################################################################
# Reference microphone calibration
################################################################################
class ReferenceCalibrationSettings(ReferenceSettingsMixin, BaseChirpSettings):
    pass


class ReferenceCalibrationController(ReferenceControllerMixin,
                                     BaseChirpController):

    def finalize(self):
        ref_mic_sens = self.get_current_value('ref_mic_sens')
        ref_mic_gain = dbi(self.get_current_value('ref_mic_gain'))
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=[exp_mic_gain, ref_mic_gain])

        exp_mic_psd, ref_mic_psd = db(results['mic_psd'])
        exp_mic_waveform, ref_mic_waveform = \
            results['mic_waveforms'].mean(axis=0)
        results['exp_mic_waveform'] = exp_mic_waveform
        results['ref_mic_waveform'] = ref_mic_waveform
        results['ref_mic_psd'] = ref_mic_psd
        results['exp_mic_psd'] = exp_mic_psd
        results['exp_mic_sens'] = exp_mic_psd+db(ref_mic_sens)-ref_mic_psd
        results['speaker_spl'] = ref_mic_psd-db(ref_mic_sens)-db(20e-6)
        results['frequency'] = results['mic_frequency']

        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        self.model.update_plots(results, freq_lb, freq_ub)
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
# Ear transfer function
################################################################################
class HRTFSettings(HRTFSettingsMixin, BaseChirpSettings):

    exp_mic_gain = 40
    output_gain = -40
    freq_lb = 500


class HRTFController(HRTFControllerMixin, BaseChirpController):

    def finalize(self):
        exp_mic_gain = dbi(self.get_current_value('exp_mic_gain'))
        waveform_averages = self.get_current_value('waveform_averages')
        results = self.iface.process(waveform_averages=waveform_averages,
                                     input_gains=exp_mic_gain)
        exp_mic_waveform = results['mic_waveforms'].mean(axis=0)[0]
        exp_mic_psd = db(results['mic_psd'][0])

        speaker_spl = self.calibration.get_spl(results['mic_frequency'],
                                               results['mic_psd'][0])

        results['exp_mic_waveform'] = exp_mic_waveform
        results['exp_mic_psd'] = exp_mic_psd
        results['frequency'] = results['mic_frequency']
        results['speaker_spl'] = speaker_spl

        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        self.model.update_plots(results, freq_lb, freq_ub)
        self.results = results
        self.result_settings = dict(self.model.paradigm.items())
        self.complete = True


class HRTF(HRTFExperimentMixin, BaseSignalExperiment):
    pass


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
