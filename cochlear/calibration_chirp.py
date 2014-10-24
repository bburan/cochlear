from __future__ import division

import os.path
import shutil

import numpy as np
import tables

from traits.api import (HasTraits, Float, Instance, Any)
from traitsui.api import View, Item, VGroup, ToolBar, Action, HSplit
from pyface.api import ImageResource
from enable.api import Component, ComponentEditor
from chaco.api import (DataRange1D, VPlotContainer, PlotAxis, create_line_plot,
                       LogMapper, OverlayPlotContainer)

from experiment import (icon_dir, AbstractData)
from experiment.channel import FileFilteredEpochChannel
from experiment.util import get_save_file

from neurogen.util import patodb, db

from base_calibration_chirp import BaseChirpCalSettings, BaseChirpCalController
import settings


class ReferenceChirpCalSettings(BaseChirpCalSettings):

    kw = dict(context=True)
    ref_mic_sens = Float(2.0e-3, label='Ref. mic. sens. (V/Pa)', **kw)
    ref_mic_gain = Float(0, label='Ref. mic. gain (dB)', **kw)

    output_settings = VGroup(
        'output',
        'output_gain',
        'amplitude',
        label='Output settings',
        show_border=True,
    )

    mic_settings = VGroup(
        'ref_mic_sens',
        'ref_mic_gain',
        'exp_mic_gain',
        label='Microphone settings',
        show_border=True,
    )


class ReferenceChirpCalData(AbstractData):

    exp_microphone = Instance('experiment.channel.EpochChannel')
    ref_microphone = Instance('experiment.channel.EpochChannel')
    ref_mic_sens = Float()
    exp_mic_sens = Any()

    @classmethod
    def from_node(cls, node):
        obj = ReferenceChirpCalData()
        exp_node = node.exp_microphone
        ref_node = node.ref_microphone
        obj.exp_microphone = FileFilteredEpochChannel.from_node(exp_node)
        obj.ref_microphone = FileFilteredEpochChannel.from_node(ref_node)
        obj.store_node = node
        return obj

    def _create_microphone_nodes(self, fs, epoch_duration):
        if 'exp_microphone' in self.fh.root:
            self.fh.root.exp_microphone.remove()
            self.fh.root.exp_microphone_ts.remove()
            self.fh.root.ref_microphone.remove()
            self.fh.root.ref_microphone_ts.remove()
        fh = self.store_node._v_file
        filter_kw = dict(filter_freq_hp=5, filter_freq_lp=80e3,
                         filter_btype='bandpass', filter_order=1)
        node = FileFilteredEpochChannel(node=fh.root, name='exp_microphone',
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        self.exp_microphone = node
        node = FileFilteredEpochChannel(node=fh.root, name='ref_microphone',
                                        epoch_duration=epoch_duration, fs=fs,
                                        dtype=np.double, use_checksum=True,
                                        **filter_kw)
        self.ref_microphone = node

    def _create_array(self, name, array, store_node=None):
        if store_node is None:
            store_node = self.store_node
        if name in store_node:
            store_node._f_get_child(name).remove()
        return self.fh.create_array(store_node, name, array)

    def compute_transfer_functions(self, ref_mic_sens, ref_mic_gain,
                                   exp_mic_gain, waveform_averages):

        # All functions are computed using these frequencies
        frequency = self.ref_microphone.get_fftfreq()

        # Compute the PSD of each microphone in Vrms
        ref_psd = self.ref_microphone \
            .get_average_psd(waveform_averages=waveform_averages)/np.sqrt(2)
        exp_psd = self.exp_microphone \
            .get_average_psd(waveform_averages=waveform_averages)/np.sqrt(2)

        # Compensate for measurement gain settings
        ref_psd = ref_psd/(10**(ref_mic_gain/20.0))
        exp_psd = exp_psd/(10**(exp_mic_gain/20.0))

        # Actual output of speaker in pascals
        speaker_pa = ref_psd/ref_mic_sens
        speaker_spl = patodb(speaker_pa)

        # Sensitivity of experiment microphone as function of frequency
        # (Vrms/Pa)
        self.exp_mic_sens = exp_psd/speaker_pa

        self._create_array('frequency', frequency)
        self._create_array('ref_psd_vrms', ref_psd)
        self._create_array('exp_psd_vrms', exp_psd)
        self._create_array('speaker_spl', speaker_spl)
        self._create_array('exp_mic_sens', self.exp_mic_sens)


class ReferenceChirpCalController(BaseChirpCalController):

    MIC_INPUT = '{}, {}'.format(BaseChirpCalController.MIC_INPUT,
                                BaseChirpCalController.REF_MIC_INPUT)
    filename = Any

    def save(self, info=None):
        filename = get_save_file(settings.CALIBRATION_DIR,
                                 'Microphone calibration|*.mic')
        if filename is not None:
            # Ensure all data is written to file before we copy it over
            self.model.data.fh.flush()
            shutil.copy(self.filename, filename)

    def poll(self):
        waveform = self.iface_adc.read_analog(timeout=0)
        self.model.data.exp_microphone.send(waveform[0])
        self.model.data.ref_microphone.send(waveform[1])
        self.epochs_acquired += 1
        if self.epochs_acquired == self.get_current_value('averages'):
            ref_mic_sens = self.get_current_value('ref_mic_sens')
            ref_mic_gain = self.get_current_value('ref_mic_gain')
            exp_mic_gain = self.get_current_value('exp_mic_gain')
            waveform_averages = self.get_current_value('waveform_averages')
            self.model.data.compute_transfer_functions(ref_mic_sens,
                                                       ref_mic_gain,
                                                       exp_mic_gain,
                                                       waveform_averages)
            self.model.data.save(**dict(self.model.paradigm.items()))
            self.model.generate_plots()
            self.complete = True
            self.stop()


class ReferenceChirpCal(HasTraits):

    paradigm = Instance(ReferenceChirpCalSettings, ())
    data = Instance(ReferenceChirpCalData)
    container = Instance(Component)

    def generate_plots(self):
        container = VPlotContainer(padding=70, spacing=70)

        # Overlay the experiment and reference microphone signal
        overlay = OverlayPlotContainer()
        time = self.data.ref_microphone.time
        signal = self.data.ref_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='black')
        axis = PlotAxis(component=plot, orientation='left',
                        title="Ref. mic. signal (mV)")
        plot.underlays.append(axis)
        overlay.insert(0, plot)
        time = self.data.exp_microphone.time
        signal = self.data.exp_microphone.get_average()*1e3
        plot = create_line_plot((time, signal), color='red')
        axis = PlotAxis(component=plot, orientation='right',
                        title="Exp. mic. signal (mV)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Time (msec)")
        plot.underlays.append(axis)
        overlay.insert(0, plot)
        container.insert(0, overlay)

        averages = self.paradigm.waveform_averages

        index_range = DataRange1D(low_setting=self.paradigm.freq_lb*0.9,
                                  high_setting=self.paradigm.freq_ub*1.1)
        index_mapper = LogMapper(range=index_range)

        # Overlay the experiment and reference microphone response (FFT)
        frequency = self.data.ref_microphone.get_fftfreq()
        ref_psd_vrms = self.data.ref_microphone \
            .get_average_psd(waveform_averages=averages)/np.sqrt(2)
        exp_psd_vrms = self.data.exp_microphone \
            .get_average_psd(waveform_averages=averages)/np.sqrt(2)
        ref_plot = create_line_plot((frequency[1:], db(ref_psd_vrms[1:], 1e-3)),
                                    color='black')
        exp_plot = create_line_plot((frequency[1:], db(exp_psd_vrms[1:], 1e-3)),
                                    color='red')
        ref_plot.index_mapper = index_mapper
        exp_plot.index_mapper = index_mapper
        axis = PlotAxis(component=exp_plot, orientation='bottom',
                        title='Frequency (Hz)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=exp_plot, orientation='right',
                        title='Exp. mic. resp (dB re 1mV)')
        exp_plot.underlays.append(axis)
        axis = PlotAxis(component=ref_plot, orientation='left',
                        title='Ref. mic. resp (dB re 1mV)')
        ref_plot.underlays.append(axis)
        overlay = OverlayPlotContainer(ref_plot, exp_plot)
        container.insert(0, overlay)

        # Convert the reference microphone response to speaker output
        ref_psd_pa = ref_psd_vrms/self.paradigm.ref_mic_sens
        ref_psd_spl = patodb(ref_psd_pa)
        plot = create_line_plot((frequency[1:], ref_psd_spl[1:]), color='black')
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Speaker output (dB SPL)")
        plot.underlays.append(axis)
        axis = PlotAxis(component=plot, orientation='bottom',
                        title="Frequency (Hz)")
        plot.underlays.append(axis)
        container.insert(0, plot)

        plot = create_line_plot((frequency, db(self.data.exp_mic_sens, 1e-3)))
        plot.index_mapper = index_mapper
        axis = PlotAxis(component=plot, orientation='left',
                        title="Exp. mic. sens mV (dB re Pa)")
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
            Action(name='Save', action='save',
                   image=ImageResource('document_save', icon_dir),
                   enabled_when='handler.complete')
        ),
        resizable=True,
    )


def launch_gui(output='ao0', **kwargs):
    tempfile = os.path.join(settings.TEMP_DIR, 'temp_mic.cal')
    with tables.open_file(tempfile, 'w') as fh:
        data = ReferenceChirpCalData(store_node=fh.root)
        controller = ReferenceChirpCalController(filename=tempfile)
        paradigm = ReferenceChirpCalSettings(output=output)
        ReferenceChirpCal(data=data, paradigm=paradigm) \
            .edit_traits(handler=controller, **kwargs)
