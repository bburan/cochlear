import os.path
from glob import glob

ROOT_DIR = os.path.join(r'c:\\', 'data', 'cochlear')
CALIBRATION_DIR = os.path.join(ROOT_DIR, 'calibration')
SETTINGS_DIR = os.path.join(ROOT_DIR, 'settings')
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')
DATA_DIR = os.path.join(ROOT_DIR, 'animals')


def list_mic_cal():
    return glob(os.path.join(CALIBRATION_DIR, '*.mic'))
