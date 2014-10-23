from __future__ import division

import shutil

import numpy as np
import tables

from traits.api import (HasTraits, Instance, Int, Any, Bool)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper)
from chaco.tools.api import BetterSelectingZoom

from experiment import (icon_dir, AbstractController)
from experiment.channel import FileFilteredEpochChannel
from experiment.util import get_save_file

from neurogen import block_definitions as blocks
from neurogen.calibration import SimpleCalibration
from neurogen.util import db

from nidaqmx import (DAQmxDefaults, TriggeredDAQmxSource, DAQmxSink,
                     DAQmxAttenControl)

from calibration_chirp import ChirpCalSettings, ChirpCalData

DAC_FS = 200e3
ADC_FS = 200e3


class ChirpCalData(ChirpCalData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    frequency = Any()
    speaker_spl = Any()

    def _create_microphone_nodes(self, fs, epoch_duration):
        if 'exp_microphone' in self.fh.root:
            self.fh.root.exp_microphone.remove()
            self.fh.root.exp_microphone_ts.remove()
        fh = self.store_node._v_file
        filter_kw = dict(filter_freq_hp=1000, filter_freq_lp=50e3,
                         filter_btype='bandpass', filter_order=1)
        node = FileFilteredEpochChannel(node=fh.root, name='exp_microphone',
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        self.exp_microphone = node

    def compute_transfer_functions(self, mic_cal, exp_mic_gain,
                                   waveform_averages):

        # All functions are computed using these frequencies
        frequency = self.exp_microphone.get_fftfreq()

        # Compute the PSD of microphone in Vrms and compensate for measurement
        # gain setting
        exp_psd = self.exp_microphone \
            .get_average_psd(waveform_averages=waveform_averages)
        exp_psd_vrms = exp_psd/np.sqrt(2)/(10**(exp_mic_gain/20.0))
        speaker_spl = mic_cal.get_spl(frequency, exp_psd_vrms)
        self._create_array('frequency', frequency)
        self._create_array('exp_psd_vrms', exp_psd_vrms)
        self._create_array('speaker_spl', speaker_spl)
        self.frequency = frequency
        self.speaker_spl = speaker_spl


class ChirpCalController(DAQmxDefaults, AbstractController):

    adc_fs = ADC_FS
    dac_fs = DAC_FS
    epochs_acquired = Int(0)
    complete = Bool(False)
    calibration_accepted = Bool(False)
    fh = Any(None)

    mic_cal = Any(None)

    def save(self, info=None):
        filename = get_save_file('c:/', 'Speaker calibration|*.spk')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.fh.flush()
            shutil.copy(self.filename, filename)

    def start(self, info=None):
        self.complete = False
        self.state = 'running'
        if self.fh is not None:
            self.fh.close()
        self.initialize_context()
        self.refresh_context()

        self.epochs_acquired = 0
        self.complete = False
        self.calibration_accepted = False

        output = self.get_current_value('output')
        duration = self.get_current_value('duration')
        ici = self.get_current_value('ici')
        epoch_duration = duration
        averages = self.get_current_value('averages')
        output_gain = self.get_current_value('output_gain')
        vrms = self.get_current_value('amplitude')
        calibration = SimpleCalibration.as_attenuation(vrms=vrms)

        analog_output = '/{}/{}'.format(self.DEV, output)

        self.iface_adc = TriggeredDAQmxSource(fs=self.adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=self.MIC_INPUT,
                                              counter_line=self.AI_COUNTER,
                                              trigger_line=self.AI_TRIGGER,
                                              callback=self.poll,
                                              trigger_duration=10e-3)

        self.iface_att = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
                                           cs_line=self.VOLUME_CS,
                                           data_line=self.VOLUME_SDI,
                                           mute_line=self.VOLUME_MUTE,
                                           zc_line=self.VOLUME_ZC,
                                           hw_clock=self.DIO_CLOCK)
        self.iface_dac = DAQmxSink(name='sink',
                                   fs=self.dac_fs,
                                   calibration=calibration,
                                   output_line=analog_output,
                                   trigger_line=self.SPEAKER_TRIGGER,
                                   run_line=self.SPEAKER_RUN,
                                   fixed_attenuation=True,
                                   hw_attenuation=0)

        # By using an Attenuation calibration and setting tone level to 0, a
        # sine wave at the given amplitude (as specified in the settings) will
        # be generated at each frequency as the reference.
        ramp = blocks.LinearRamp(name='sweep')
        graph = blocks.Tone(name='tone', level=0, frequency=ramp) >> \
            blocks.Cos2Envelope(name='envelope', rise_time=2.5e-3) >> \
            self.iface_dac
        self.current_graph = graph

        self.current_graph.set_value('sweep.ramp_duration', duration)
        self.current_graph.set_value('envelope.duration', duration)
        self.current_graph.set_value('sink.duration', epoch_duration)
        self.model.data._create_microphone_nodes(self.adc_fs, epoch_duration)

        freq_lb = self.get_current_value('freq_lb')
        self.current_graph.set_value('sweep.start', freq_lb)

        freq_ub = self.get_current_value('freq_ub')
        self.current_graph.set_value('sweep.stop', freq_ub)

        self.current_graph.queue_init('FIFO')
        self.current_graph.queue_append(averages, ici)

        # Initialize and reserve the NIDAQmx hardware
        self.iface_adc.setup()
        self.iface_att.setup()

        self.iface_att.set_mute(False)
        self.iface_att.set_zero_crossing(False)
        if output == 'ao0':
            self.iface_att.set_gain(-np.inf, output_gain)
        elif output == 'ao1':
            self.iface_att.set_gain(output_gain, -np.inf)

        # Need to clear this so that the signal output can take over.
        self.iface_att.clear()
        self.iface_adc.start()
        self.current_graph.play_queue()

    def stop(self, info=None):
        self.state = 'halted'
        self.iface_dac.clear()
        self.iface_adc.clear()
        self.complete = True

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.exp_microphone.send(waveform)
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            exp_mic_gain = self.get_current_value('exp_mic_gain')
            waveform_averages = self.get_current_value('waveform_averages')
            self.model.data.compute_transfer_functions(self.mic_cal,
                                                       exp_mic_gain,
                                                       waveform_averages)
            self.model.generate_plots()
            self.complete = True
            self.stop()

    def update_inear(self, info):
        self.calibration_accepted = True
        info.ui.dispose()

    def cancel_inear(self, info):
        self.calibration_accepted = False
        info.ui.dispose()


class ChirpCal(HasTraits):

    paradigm = Instance(ChirpCalSettings, ())
    data = Instance(ChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        # Overlay the experiment and reference microphone signal
        time = self.data.exp_microphone.time
        signal = self.data.exp_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. signal (mV)")
        plot.underlays.append(axis)
        tool = BetterSelectingZoom(component=plot)
        plot.tools.append(tool)
        container.insert(0, plot)

        averages = self.paradigm.waveform_averages
        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        # Overlay the experiment and reference microphone response (FFT)
        frequency = self.data.exp_microphone.get_fftfreq()
        exp_psd_vrms = self.data.exp_microphone \
            .get_average_psd(waveform_averages=averages)/np.sqrt(2)
        exp_plot = create_line_plot((frequency[1:], db(exp_psd_vrms[1:], 1e-3)),
                                    color='black')
        exp_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title='Frequency (Hz)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='left',
                        title='Exp. mic. resp (dB re 1mV)')
        exp_plot.underlays.append(axis)
        container.insert(0, exp_plot)

        plot = create_line_plot((frequency[1:],
                                 self.data.speaker_spl[1:]),
                                color='black')
        plot.index_mapper = index_mapper
        plot.value_mapper.low_pos = 0
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker output (dB SPL)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        self.container = container

    traits_view = View(
        HSplit(
            Item('paradigm', style='custom', width=200,
                 enabled_when='handler.state!="running"'),
            VGroup(
                Item('handler.epochs_acquired', style='readonly'),
                VGroup(
                    Item('container', editor=ComponentEditor(), width=500,
                         height=800, show_label=False),
                ),
            ),
            show_labels=False,
        ),
        toolbar=ToolBar(
            '-',
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='not handler.state=="running"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('Stop', icon_dir),
                   enabled_when='handler.state=="running"'),
            '-',
            Action(name='Accept', action='update_inear',
                   image=ImageResource('dialog_ok_apply', icon_dir),
                   enabled_when='handler.complete'),
            Action(name='Cancel', action='cancel_inear',
                   image=ImageResource('dialog_cancel', icon_dir),
                   enabled_when='handler.complete'),

        ),
        resizable=True,
    )


def launch_gui(mic_cal, **kwargs):
    with tables.open_file('temp.hdf5', 'w') as fh:
        data = ChirpCalData(store_node=fh.root)
        controller = ChirpCalController(mic_cal=mic_cal)
        experiment = ChirpCal(data=data)
        experiment.edit_traits(handler=controller, **kwargs)
        if not controller.calibration_accepted:
            return None
        frequency = experiment.data.frequency
        magnitude = experiment.data.speaker_spl
        phase = np.zeros_like(magnitude)
        vrms = experiment.paradigm.amplitude
        gain = experiment.paradigm.output_gain
        return SimpleCalibration.from_single_vrms(frequency, magnitude, phase,
                                                  vrms, gain=gain)


def main(mic_cal):
    controller = ChirpCalController(mic_cal=mic_cal)
    ChirpCal().configure_traits(handler=controller)


if __name__ == '__main__':
    filename = 'C:/Users/bburan/Desktop/141017 acoustic system calibration.mic'
    with tables.open_file(filename, 'r') as fh:
        frequency = fh.root.frequency.read()
        exp_mic_sens = fh.root.exp_mic_sens.read()
        mic_cal = SimpleCalibration.from_mic_sens(frequency, exp_mic_sens)
        main(mic_cal)
