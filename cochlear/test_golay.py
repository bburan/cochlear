from neurogen import block_definitions as blocks
from cochlear import nidaqmx as ni
from neurogen.calibration import util

def main():
    a, b = util.golay_pair(16)
