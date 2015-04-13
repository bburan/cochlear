import os.path
import datetime as dt
import re

import numpy as np
import tables

from traits.api import (HasTraits, Property, List, Str, Instance, Date, Enum,
                        Event)
from traitsui.api import (View, VGroup, EnumEditor, Item, ToolBar, Action,
                          Controller, HGroup)
from pyface.api import ImageResource

from experiment import icon_dir
from neurogen.calibration import InterpCalibration

from cochlear import settings
from cochlear import calibration_chirp as calibration
from cochlear import abr_experiment
from cochlear import dpoae_experiment


class ExperimentController(Controller):

    def run_microphone_calibration(self, info):
        calibration.launch_gui('ao0', parent=info.ui.control, kind='livemodal')
        info.object._update_calibrations()

    def _get_filename(self, info, experiment):
        datetime = dt.datetime.now()
        filename = info.object.base_filename.format(
            date=datetime.strftime('%Y%m%d'),
            time=datetime.strftime('%H%M'),
            experiment=experiment)
        return os.path.join(settings.DATA_DIR, filename)

    def run_abr_experiment(self, info):
        filename = self._get_filename(info, 'ABR')
        abr_experiment.launch_gui(info.object.mic_cal, filename=filename,
                                  parent=info.ui.control, kind='livemodal')

    def run_abr_check(self, info):
        paradigm_dict = {
            'averages': 128,
            'repetition_rate': 40,
            'level': 'exact_order([80], c=1)',
            'frequency': '11310',
            'reject_threshold': 1.0,
        }
        abr_experiment.launch_gui(info.object.mic_cal,
                                  filename=None,
                                  paradigm_dict=paradigm_dict,
                                  parent=info.ui.control,
                                  kind='livemodal')

    def run_dpoae_experiment(self, info):
        filename = self._get_filename(info, 'DPOAE')
        dpoae_experiment.launch_gui(info.object.mic_cal, filename=filename,
                                    parent=info.ui.control, kind='livemodal')

    def run_dpoae_check(self, info):
        paradigm_dict = {
            'f2_frequency': 8e3,
            'f2_level': 'exact_order(np.arange(40, 95, 10), c=1)'
        }
        dpoae_experiment.launch_gui(info.object.mic_cal,
                                    filename=None,
                                    paradigm_dict=paradigm_dict,
                                    parent=info.ui.control,
                                    kind='livemodal')


class ExperimentSetup(HasTraits):

    experimenters = Property(List)
    experimenter = Str
    experiment_note = Str

    animals = Property(List)
    animal = Str
    ear = Enum(('right', 'left'))

    calibrations = List
    calibration = Str

    mic_cal = Instance('neurogen.calibration.Calibration')

    base_filename = Property(depends_on='experimenter, animal, experiment_note')

    # Events that can be logged
    reposition_pt_event = Event
    administer_anesthesia_event = Event

    def _date_default(self):
        return dt.date.today()

    def _get_base_filename(self):
        t = '{{date}}-{{time}} {} {} {} {} {{experiment}}.hdf5'
        f = t.format(self.experimenter, self.animal, self.ear,
                     self.experiment_note)
        return re.sub(r'\s+', r' ', f)

    def _calibrations_default(self):
        return self._update_calibrations()

    def _update_calibrations(self):
        calibrations = settings.list_mic_cal()
        calibrations = [os.path.basename(c) for c in calibrations]
        self.calibration = calibrations[-1]
        return calibrations

    def _get_experimenters(self):
        return ['Brad', 'Stephen']

    def _get_animals(self):
        return ['Beowulf', 'Tarragon', 'Dill', 'Parsley', 'Cumin', 'Zeera',
                'Gamun']

    def _calibration_changed(self, new):
        filename = os.path.join(settings.CALIBRATION_DIR, new)
        self.mic_cal = InterpCalibration.from_mic_file(filename)

    view = View(
        VGroup(
            Item('experimenter', editor=EnumEditor(name='experimenters')),
            HGroup(
                Item('animal', editor=EnumEditor(name='animals')),
                Item('ear'),
            ),
            Item('experiment_note'),
            Item('calibration', editor=EnumEditor(name='calibrations')),
            Item('base_filename', style='readonly'),
            show_border=True,
        ),
        resizable=True,
        toolbar=ToolBar(
            Action(name='Mic cal',
                   image=ImageResource('media_record', icon_dir),
                   action='run_microphone_calibration'),
            Action(name='ABR',
                   image=ImageResource('view_statistics', icon_dir),
                   enabled_when='mic_cal is not None '
                                'and animal '
                                'and experimenter ',
                   action='run_abr_experiment'),
            Action(name='ABR check',
                   image=ImageResource('view_statistics', icon_dir),
                   enabled_when='mic_cal is not None', action='run_abr_check'),
            Action(name='DPOAE',
                   image=ImageResource('datashowchart', icon_dir),
                   enabled_when='mic_cal is not None '
                                'and animal '
                                'and experimenter ',
                   action='run_dpoae_experiment'),
            Action(name='DPOAE check',
                   image=ImageResource('datashowchart', icon_dir),
                   enabled_when='mic_cal is not None',
                   action='run_dpoae_check'),
        ),
    )


if __name__ == '__main__':
    ExperimentSetup().configure_traits(handler=ExperimentController())
