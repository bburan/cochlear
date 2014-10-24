import os.path

import tables

from traits.api import (HasTraits, Property, List, Str, Instance)
from traitsui.api import (View, VGroup, EnumEditor, Item, ToolBar, Action,
                          Controller)
from pyface.api import ImageResource

from experiment import icon_dir
from neurogen.calibration import SimpleCalibration

from cochlear import settings
from cochlear import calibration_chirp as cal_mic
from cochlear import calibration_inear as cal_inear
from cochlear import abr_experiment
from cochlear import dpoae_experiment


class ExperimentController(Controller):

    def run_microphone_calibration(self, info):
        cal_mic.launch_gui(parent=info.ui.control, kind='livemodal')
        info.object._update_calibrations()

    def run_inear_cal_0(self, info):
        calibration = cal_inear.launch_gui('ao0', info.object.mic_calibration,
                                           parent=info.ui.control,
                                           kind='livemodal')
        if calibration is not None:
            info.object.inear_cal_0 = calibration

    def run_inear_cal_1(self, info):
        calibration = cal_inear.launch_gui('ao1', info.object.mic_calibration,
                                           parent=info.ui.control,
                                           kind='livemodal')
        if calibration is not None:
            info.object.inear_cal_1 = calibration

    def run_abr_experiment(self, info):
        abr_experiment.launch_gui(info.object.inear_cal_0,
                                  parent=info.ui.control,
                                  kind='livemodal')

    def run_dpoae_experiment(self, info):
        dpoae_experiment.launch_gui(info.object.inear_cal_0,
                                    info.object.inear_cal_1,
                                    info.object.mic_calibration,
                                    parent=info.ui.control, kind='livemodal')


class ExperimentSetup(HasTraits):

    experimenters = Property(List)
    experimenter = Str

    animals = Property(List)
    animal = Str

    calibrations = List
    calibration = Str

    mic_calibration = Instance('neurogen.calibration.SimpleCalibration')
    inear_cal_0 = Instance('neurogen.calibration.SimpleCalibration')
    inear_cal_1 = Instance('neurogen.calibration.SimpleCalibration')

    def _calibrations_default(self):
        return self._update_calibrations()

    def _update_calibrations(self):
        calibrations = settings.list_mic_calibrations()
        calibrations = [os.path.basename(c) for c in calibrations]
        self.calibration = calibrations[0]
        return calibrations

    def _get_experimenters(self):
        return ['Test', 'Brad', 'Stephen']

    def _get_animals(self):
        return ['Test', 'Oyster', 'Truffle']

    def _animal_changed(self):
        self.inear_calibration = None

    def _calibration_changed(self, new):
        filename = os.path.join(settings.CALIBRATION_DIR, new)
        with tables.open_file(filename, 'r') as fh:
            frequency = fh.root.frequency.read()
            exp_mic_sens = fh.root.exp_mic_sens.read()
            cal = SimpleCalibration.from_mic_sens(frequency, exp_mic_sens)
            self.mic_calibration = cal

    view = View(
        VGroup(
            Item('experimenter', editor=EnumEditor(name='experimenters')),
            Item('animal', editor=EnumEditor(name='animals')),
            Item('calibration', editor=EnumEditor(name='calibrations')),
            show_border=True,
        ),
        resizable=True,
        toolbar=ToolBar(
            Action(name='Mic cal',
                   image=ImageResource('media_record', icon_dir),
                   action='run_microphone_calibration'),
            Action(name='Left cal',
                   image=ImageResource('speaker', icon_dir),
                   enabled_when='mic_calibration is not None and '
                                'animal is not None and '
                                'experimenter is not None',
                   action='run_inear_cal_0'),
            Action(name='Right cal',
                   image=ImageResource('speaker', icon_dir),
                   enabled_when='mic_calibration is not None and '
                                'animal is not None and '
                                'experimenter is not None',
                   action='run_inear_cal_1'),
            Action(name='ABR',
                   image=ImageResource('view_statistics', icon_dir),
                   enabled_when='inear_cal_0 is not None',
                   action='run_abr_experiment'),
            Action(name='DPOAE',
                   image=ImageResource('datashowchart', icon_dir),
                   enabled_when='inear_cal_0 is not None and '
                                'inear_cal_1 is not None',
                   action='run_dpoae_experiment'),
        ),
    )


if __name__ == '__main__':
    ExperimentSetup().configure_traits(handler=ExperimentController())
