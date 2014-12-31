from __future__ import division

import shutil

import numpy as np

from traits.api import (Float, Int, Property, Any, Bool, Enum)
from traitsui.api import View, Item, VGroup, Include
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment import (icon_dir, AbstractController, AbstractParadigm)
from experiment.util import get_save_file

from neurogen import block_definitions as blocks
from neurogen.calibration import InterpCalibration
from neurogen.util import patodb, db

from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, QueuedDAQmxPlayer,
                     DAQmxChannel, DAQmxAttenControl)

DAC_FS = 200e3
ADC_FS = 200e3


def get_chirp_transform(vrms):
    calibration_data = np.array([
        (1, 30),
        (100e3, -10),
        #(100, 0),
        #(100e3, 0),
    ])
    frequencies = calibration_data[:, 0]
    magnitude = calibration_data[:, 1]
    return InterpCalibration.from_single_vrms(frequencies, magnitude, vrms)


class BaseChirpCalSettings(AbstractParadigm):

    kw = dict(context=True)
    exp_mic_gain = Float(20, label='Exp. mic. gain (dB)', **kw)

    output = Enum(('ao1', 'ao0'), label='Analog output (channel)', **kw)
    rise_time = Float(5e-3, label='Envelope rise time', **kw)
    amplitude = Float(1, label='Waveform amplitude (Vrms)', **kw)
    output_gain = Float(6, label='Output gain (dB)', **kw)
    freq_lb = Float(0.1e3, label='Start frequency (Hz)', **kw)
    freq_ub = Float(60e3, label='End frequency (Hz)', **kw)
    freq_resolution = Float(50, label='Frequency resolution (Hz)')
    fft_averages = Int(32, label='Number of FFTs', **kw)
    waveform_averages = Int(32, label='Number of chirps per FFT', **kw)
    ici = Float(0.01, label='Inter-chirp interval', **kw)

    averages = Property(depends_on='fft_averages, waveform_averages',
                        label='Number of chirps', **kw)
    duration = Property(depends_on='freq_resolution',
                        label='Chirp duration (sec)', **kw)
    total_duration = Property(depends_on='duration, averages, ici',
                              label='Total duration (sec)')

    def _get_duration(self):
        return 1/self.freq_resolution

    def _get_total_duration(self):
        return (self.duration+self.ici)*self.averages

    def _get_averages(self):
        return self.fft_averages*self.waveform_averages

    output_settings = VGroup(
        Item('output', style='readonly'),
        'output_gain',
        'amplitude',
        label='Output settings',
        show_border=True,
    )

    stimulus_settings = VGroup(
        'rise_time',
        'freq_lb',
        'freq_ub',
        'freq_resolution',
        'fft_averages',
        'waveform_averages',
        'ici',
        Item('averages', style='readonly'),
        Item('duration', style='readonly'),
        Item('total_duration', style='readonly'),
        show_border=True,
        label='Chirp settings',
    )

    mic_settings = VGroup(
        'exp_mic_gain',
        label='Microphone settings',
        show_border=True,
    )

    traits_view = View(
        VGroup(
            Include('output_settings'),
            Include('mic_settings'),
            Include('stimulus_settings'),
        )
    )


class BaseChirpCalController(DAQmxDefaults, AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)
    complete = Bool(False)
    fh = Any(None)

    def save(self, info=None):
        filename = get_save_file('c:/', 'Speaker calibration|*.spk')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.fh.flush()
            shutil.copy(self.filename, filename)

    def _setup_input(self):
        epoch_duration = self.get_current_value('duration')
        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.MIC_INPUT,
                                              callback=self.poll,
                                              trigger_duration=10e-3,
                                              expected_range=1)
        self.model.data._create_microphone_nodes(self.adc_fs, epoch_duration)
        self.iface_adc.setup()
        self.iface_adc.start()

    def _setup_output(self, output=None):
        if output is None:
            output = self.get_current_value('output')
        freq_lb = self.get_current_value('freq_lb')
        freq_ub = self.get_current_value('freq_ub')
        epoch_duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        averages = self.get_current_value('averages')
        output_gain = self.get_current_value('output_gain')
        vrms = self.get_current_value('amplitude')
        rise_time = self.get_current_value('rise_time')
        calibration = get_chirp_transform(vrms)
        analog_output = '/{}/{}'.format(self.DEV, output)

        self.iface_att = DAQmxAttenControl()
        self.iface_dac = QueuedDAQmxPlayer(fs=self.dac_fs,
                                           output_line=analog_output,
                                           duration=epoch_duration)

        # By using an Attenuation calibration and setting tone level to 0, a
        # sine wave at the given amplitude (as specified in the settings) will
        # be generated at each frequency as the reference.
        ramp = blocks.LinearRamp(name='sweep')
        self.current_channel = \
            blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope') >> \
            DAQmxChannel(calibration=calibration)
        self.current_channel.set_value('sweep.ramp_duration', epoch_duration)
        self.current_channel.set_value('envelope.duration', epoch_duration)
        self.current_channel.set_value('envelope.rise_time', rise_time)
        self.current_channel.set_value('sweep.start', freq_lb)
        self.current_channel.set_value('sweep.stop', freq_ub)

        self.iface_dac.add_channel(self.current_channel)
        self.iface_dac.queue_init('FIFO')
        self.iface_dac.queue_append(averages, ici)

        # Need to setup, configure attenuation and then clear the task so that
        # the signal output can take over port 0.
        self.iface_att.setup()
        if output == 'ao0':
            self.iface_att.set_gain(-np.inf, output_gain)
        elif output == 'ao1':
            self.iface_att.set_gain(output_gain, -np.inf)
        self.iface_att.clear()

    def start(self, info=None):
        self.complete = False
        self.state = 'running'
        self.epochs_acquired = 0
        self.complete = False
        self.calibration_accepted = False
        if self.fh is not None:
            self.fh.close()

        self.initialize_context()
        self.refresh_context()
        self._setup_input()
        self._setup_output()
        self.iface_dac.play_queue()

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.complete = True
