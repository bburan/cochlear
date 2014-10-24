from __future__ import division

import os.path

import numpy as np
import tables

from traits.api import (HasTraits, Instance, Any, Bool)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper)
from chaco.tools.api import BetterSelectingZoom

from experiment.channel import FileFilteredEpochChannel
from experiment import AbstractData, icon_dir

from neurogen.calibration import SimpleCalibration
from neurogen.util import db

from base_calibration_chirp import BaseChirpCalSettings, BaseChirpCalController
import settings


class InearChirpCalSettings(BaseChirpCalSettings):

    amplitude = 1
    fft_averages = 8
    waveform_averages = 8
    frequency_resolution = 50


class InearChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    frequency = Any()
    speaker_spl = Any()
    speaker_spl_vrms = Any()

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

    def compute_transfer_functions(self, mic_cal, output_cal, exp_mic_gain,
                                   waveform_averages):

        # All functions are computed using these frequencies
        frequency = self.exp_microphone.get_fftfreq()
        output_vrms = output_cal.get_sf(frequency, level=0, attenuation=0)

        # Compute the PSD of microphone in Vrms and compensate for measurement
        # gain setting
        exp_psd = self.exp_microphone \
            .get_average_psd(waveform_averages=waveform_averages)
        exp_psd_vrms = exp_psd/np.sqrt(2)/(10**(exp_mic_gain/20.0))
        speaker_spl = mic_cal.get_spl(frequency, exp_psd_vrms)
        speaker_spl_vrms = mic_cal.get_spl(frequency, exp_psd_vrms/output_vrms)
        self._create_array('frequency', frequency)
        self._create_array('exp_psd_vrms', exp_psd_vrms)
        self._create_array('speaker_spl', speaker_spl)
        self._create_array('speaker_spl_vrms', speaker_spl)
        self.frequency = frequency
        self.speaker_spl = speaker_spl
        self.speaker_spl_vrms = speaker_spl_vrms

    def _create_array(self, name, array, store_node=None):
        if store_node is None:
            store_node = self.store_node
        if name in store_node:
            store_node._f_get_child(name).remove()
        return self.fh.create_array(store_node, name, array)


class InearChirpCalController(BaseChirpCalController):

    mic_cal = Any(None)
    calibration_accepted = Bool(False)

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.exp_microphone.send(waveform)
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            exp_mic_gain = self.get_current_value('exp_mic_gain')
            waveform_averages = self.get_current_value('waveform_averages')
            self.model.data.compute_transfer_functions(self.mic_cal,
                                                       self.current_channel.calibration,
                                                       exp_mic_gain,
                                                       waveform_averages)
            self.model.data.save(**dict(self.model.paradigm.items()))
            self.model.generate_plots()
            self.complete = True
            self.stop()

    def update_inear(self, info):
        self.calibration_accepted = True
        info.ui.dispose()

    def cancel_inear(self, info):
        self.calibration_accepted = False
        info.ui.dispose()


class InearChirpCal(HasTraits):

    paradigm = Instance(InearChirpCalSettings, ())
    data = Instance(InearChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        # Overlay the experiment and reference microphone signal
        time = self.data.exp_microphone.time*1e3
        signal = self.data.exp_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. signal (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (msec)")
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
                                value_bounds=(40, self.data.speaker_spl.max()),
                                color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Actual speaker output (dB SPL)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        plot = create_line_plot((frequency[1:],
                                 self.data.speaker_spl_vrms[1:]),
                                value_bounds=(40, self.data.speaker_spl_vrms.max()),
                                color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker output at 1Vrms (dB SPL)")
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


def launch_gui(output, mic_cal, **kwargs):
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_inear.cal')
    with tables.open_file(tempfile, 'w') as fh:
        data = InearChirpCalData(store_node=fh.root)
        controller = InearChirpCalController(mic_cal=mic_cal)
        paradigm = InearChirpCalSettings(output=output)
        experiment = InearChirpCal(data=data, paradigm=paradigm)
        experiment.edit_traits(handler=controller, **kwargs)
        if not controller.calibration_accepted:
            return None
        frequency = experiment.data.frequency
        magnitude = experiment.data.speaker_spl_vrms
        phase = np.zeros_like(magnitude)
        gain = experiment.paradigm.output_gain
        return SimpleCalibration.from_single_vrms(frequency, magnitude, phase,
                                                  gain=gain)
