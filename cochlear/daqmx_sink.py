import nidaqmx as daq
import time
from neurogen import block_definitions as blocks
from neurogen.calibration import Attenuation


def test_neurogen():
    #ai0 = daq.TriggeredDAQmxSource()
    #ai0.setup()
    graph = blocks.Tone(level=-10) >> \
        blocks.Cos2Envelope(duration=10, rise_time=0.5e-3) >> \
        daq.DAQmxSink(calibration=Attenuation(), duration=10)
    graph.play_continuous(10)
    #graph.join()
    #print ai0.samples_available()


def test_atten():
    #import PyDAQmx as ni
    #import ctypes
    #task = ni.TaskHandle(0)
    #ni.DAQmxCreateTask(name, ctypes.byref(task))
    #ni.DAQmxCreateAOVoltageChan(task, output_line, '', -5, 5,
    #                            ni.DAQmx_Val_Volts, '')
    iface = daq.DAQmxAttenControl()
    iface.setup_volume()
    #gain = -95.5
    gain = 31.5
    iface.set_gain(gain, gain)
    time.sleep(0.1)
    iface.clear()
    #return iface


if __name__ == '__main__':
    test_atten()
    test_neurogen()
