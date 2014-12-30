import os.path
from glob import glob

DATA_DIR = os.path.join(r'c:\\', 'data', 'cochlear')
CALIBRATION_DIR = os.path.join(DATA_DIR, 'calibration')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')


def list_mic_cal():
    return glob(os.path.join(CALIBRATION_DIR, '*.mic'))
