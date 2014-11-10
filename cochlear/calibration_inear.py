from __future__ import division

import os.path

import numpy as np
import tables
from scipy import signal

from traits.api import (HasTraits, Instance, Any, Bool)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment.channel import FileFilteredEpochChannel
from experiment import AbstractData, icon_dir

from neurogen.calibration import LinearCalibration
from neurogen.util import db

from base_calibration_chirp import BaseChirpCalSettings, BaseChirpCalController
import settings


class InearChirpCalSettings(BaseChirpCalSettings):

    fft_averages = 4
    waveform_averages = 8
    frequency_resolution = 50


class InearChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')

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

    def compute_transfer_functions(self, exp_mic_sens, waveform_fs, waveform,
                                   exp_mic_gain, waveform_averages):

        # All functions are computed using these frequencies
        self.frequency = self.exp_microphone.get_fftfreq()

        # Compute the PSD of microphone in Vrms and compensate for measurement
        # gain setting
        exp_psd_rms = self.exp_microphone \
            .get_average_psd(waveform_averages=waveform_averages, rms=True)
        self.exp_psd_rms = db(exp_psd_rms)-exp_mic_gain

        psd = 2*np.abs(np.fft.rfft(waveform))/len(waveform)
        self.dac_psd_rms = db(psd/np.sqrt(2))
        self.dac_frequency = np.fft.rfftfreq(len(waveform), 1/waveform_fs)

        self.speaker_sens = self.dac_psd_rms+exp_mic_sens-self.exp_psd_rms
        self.speaker_spl = db(1)-self.speaker_sens-db(20e-6)

        self._create_array('frequency', self.frequency)
        self._create_array('exp_psd_rms', self.exp_psd_rms)
        self._create_array('dac_frequency', self.dac_frequency)
        self._create_array('dac_psd_rms', self.dac_psd_rms)
        self._create_array('speaker_sens', self.speaker_sens)

    def _create_array(self, name, array, store_node=None):
        if store_node is None:
            store_node = self.store_node
        if name in store_node:
            store_node._f_get_child(name).remove()
        return self.fh.create_array(store_node, name, array)


class InearChirpCalController(BaseChirpCalController):

    calibration_accepted = Bool(False)

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.exp_microphone.send(waveform)
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            self.finalize()

    def finalize(self):
        exp_mic_gain = self.get_current_value('exp_mic_gain')
        waveform_averages = self.get_current_value('waveform_averages')
        waveform_fs, waveform = \
            self.iface_dac.fs, self.iface_dac.realize().ravel()
        self.model.data.compute_transfer_functions(self.model.exp_mic_sens,
                                                   waveform_fs, waveform,
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
    exp_mic_sens = Any(None)

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        # Overlay the experiment and reference microphone signal
        time = self.data.exp_microphone.time*1e3
        signal = self.data.exp_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (msec)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        # Overlay the experiment and reference microphone response (FFT)
        frequency = self.data.frequency[1:]
        exp_mic_sens = self.exp_mic_sens[1:]
        exp_psd_rms = self.data.exp_psd_rms[1:]
        dac_psd_rms = self.data.dac_psd_rms[1:]
        speaker_spl = self.data.speaker_spl[1:]

        overlay = OverlayPlotContainer()
        exp_plot = create_line_plot((frequency, exp_psd_rms), color='black')
        exp_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title='Frequency (Hz)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='left',
                        title='Exp. mic. (dB re V)')
        exp_plot.underlays.append(axis)
        overlay.add(exp_plot)

        sens_plot = create_line_plot((frequency, exp_mic_sens), color='black')
        sens_plot.alpha = 0.5
        sens_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='right',
                        title='Exp. mic. sens (dB re V)')
        exp_plot.underlays.append(axis)
        overlay.add(sens_plot)

        container.insert(0, overlay)

        plot = create_line_plot((frequency, dac_psd_rms), color='black')
        #plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="DAC (dB re V)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        plot = create_line_plot((frequency, speaker_spl), color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker @ 1Vrms. (dB SPL)")
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


def launch_gui(output, exp_mic_sens, **kwargs):
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_inear.cal')
    with tables.open_file(tempfile, 'w') as fh:
        data = InearChirpCalData(store_node=fh.root)
        controller = InearChirpCalController()
        paradigm = InearChirpCalSettings(output=output)
        experiment = InearChirpCal(data=data, paradigm=paradigm,
                                   exp_mic_sens=exp_mic_sens)
        experiment.edit_traits(handler=controller, **kwargs)
        if not controller.calibration_accepted:
            return None
        return SimpleCalibration(experiment.data.frequency,
                                 experiment.data.speaker_sens)
