from __future__ import division

from traits.api import Instance
from traitsui.api import (View, Item, ToolBar, Action, ActionGroup, VGroup,
                          HSplit, MenuBar, Menu)
from pyface.api import ImageResource

from experiment import (AbstractParadigm, Expression, AbstractData,
                        AbstractController, AbstractExperiment, icon_dir)

from neurogen.block_definitions import (BandlimitedNoise, Cos2Envelope)
from neurogen.calibration import Attenuation
from neurogen.sink import ContinuousSink

from nidaqmx import (DAQmxDefaults, DAQmxSink, DAQmxAttenControl)


class NoiseExposureSink(DAQmxSink, ContinuousSink):

    def __init__(self, *args, **kwargs):
        DAQmxSink.__init__(self, *args, **kwargs)
        ContinuousSink.__init__(self, *args, **kwargs)


class NoiseExposureParadigm(AbstractParadigm):

    kw = dict(context=True)
    center_frequency = Expression(12e3, label='Center frequency (Hz)', **kw)
    bandwidth = Expression(8e3, label='Bandwidth (Hz)', **kw)
    level = Expression(-80, label='Level (dB SPL)', **kw)
    seed = Expression(1, label='Noise seed', **kw)
    duration = Expression('2*60*60', label='Exposure duration (sec)', **kw)


class NoiseExposureController(AbstractController, DAQmxDefaults):

    dac_fs = 100e3

    def setup_experiment(self, info=None):
        self.iface_dac = NoiseExposureSink(name='sink', fs=self.dac_fs,
                                           calibration=Attenuation(),
                                           output_line=self.SPEAKER_OUTPUT,
                                           trigger_line=self.SPEAKER_TRIGGER,
                                           run_line=self.SPEAKER_RUN)
        #self.iface_atten = DAQmxAttenControl(clock_line=self.VOLUME_CLK,
        #                                     cs_line=self.VOLUME_CS,
        #                                     data_line=self.VOLUME_SDI,
        #                                     mute_line=self.VOLUME_MUTE,
        #                                     zc_line=self.VOLUME_ZC,
        #                                     hw_clock=self.DIO_CLOCK)
        self.current_graph = BandlimitedNoise(name='noise') >> \
            Cos2Envelope(name='envelope', rise_time=10) >> \
            self.iface_dac

    def start_experiment(self, info=None):
        self.refresh_context(evaluate=True)
        # This task cannot run concurrently with the AO task because the AO task
        # uses a digital output for generating the trigger.
        #self.iface_atten.setup()
        #self.iface_atten.set_gain(0, 0)
        #self.iface_atten.clear()

        # Now, setup and run.
        self.current_graph.play_continuous()

    def set_duration(self, value):
        self.current_graph.set_value('envelope.duration', value)
        self.current_graph.set_value('sink.duration', value)

    def set_center_frequency(self, value):
        self.current_graph.set_value('noise.fc', value)

    def set_bandwidth(self, value):
        self.current_graph.set_value('noise.bandwidth', value)

    def set_level(self, value):
        self.current_graph.set_value('noise.level', value)

    def set_seed(self, value):
        self.current_graph.set_value('noise.seed', value)


class NoiseExposureExperiment(AbstractExperiment):

    paradigm = Instance(NoiseExposureParadigm, ())
    data = Instance(AbstractData, ())

    traits_view = View(
        HSplit(
            VGroup(
                Item('paradigm', style='custom', show_label=False, width=200),
                label='Paradigm',
                show_border=True,
            ),
            show_labels=False,
        ),
        resizable=True,
        height=500,
        width=800,
        toolbar=ToolBar(
            Action(name='Start', action='start',
                   image=ImageResource('1rightarrow', icon_dir),
                   enabled_when='handler.state=="uninitialized"'),
            Action(name='Stop', action='stop',
                   image=ImageResource('stop', icon_dir),
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
            Menu(
                ActionGroup(
                    Action(name='Load calibration', action='load_calibration'),
                ),
                ActionGroup(
                    Action(name='Calibrate reference microphone',
                           action='calibrate_reference_microphone'),
                    Action(name='Calibrate microphone',
                           action='calibrate_microphone'),
                    Action(name='Calibrate speaker',
                           action='calibrate_speaker'),
                ),
                name='&Calibration'
            ),

        ),
        id='lbhb.NoiseExposureExperiment',
    )


if __name__ == '__main__':
    import logging
    #logging.basicConfig(level='DEBUG')
    import PyDAQmx as ni
    ni.DAQmxResetDevice('Dev1')
    controller = NoiseExposureController()
    NoiseExposureExperiment().configure_traits(handler=controller)
