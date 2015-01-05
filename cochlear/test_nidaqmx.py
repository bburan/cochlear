import unittest
from cochlear import nidaqmx as ni
from neurogen import block_definitions as blocks
from neurogen import calibration
import PyDAQmx as _ni
import numpy as np
import time


import ctypes

class TestDAQmxConfig(unittest.TestCase):

    def setUp(self):
        time.sleep(0.1)
        _ni.DAQmxResetDevice('Dev1')
        time.sleep(0.1)
        self.fs = 200e3
        self.duration = 10e-3
        self.iti = 1e-3
        self.repetitions = 3
        self.total_duration = (self.duration+self.iti)*self.repetitions
        self.expected_size = self.repetitions, int(self.fs*self.duration)
        analog_data = np.zeros(self.fs*self.total_duration)
        run_data = np.zeros(self.fs*self.total_duration)
        trigger_data = np.zeros(self.fs*self.total_duration)
        for i in range(self.repetitions):
            lb = (self.duration+self.iti)*self.fs*i
            ub = lb+5
            trigger_data[lb:ub] = 1

        self.analog_data = analog_data
        self.run_data = run_data
        self.trigger_data = trigger_data

        # Set up iface_adc to acquire data into samples list
        self.samples = []
        iface_adc = ni.TriggeredDAQmxSource(
            ni.DAQmxDefaults.ERP_INPUT, self.fs, self.duration,
            callback=lambda s: self.samples.extend(s))
        iface_adc.start()

    def test_continuous_player(self):
        iface_dac = ni.ContinuousDAQmxPlayer(fs=self.fs)
        iface_dac.setup()
        iface_dac.write(self.analog_data, self.trigger_data, self.run_data)
        iface_dac.start()
        time.sleep(self.total_duration*1.1)
        self.assertTrue(iface_dac.complete())
        self.assertEqual(np.concatenate(self.samples).shape,
                         self.expected_size)

    def test_queued_player(self):
        iface_dac = ni.QueuedDAQmxPlayer(fs=self.fs)
        iface_dac.setup()
        iface_dac.write(self.analog_data, self.trigger_data, self.run_data)
        iface_dac.start()
        time.sleep(self.total_duration*0.6)
        self.assertFalse(iface_dac.complete())
        time.sleep(self.total_duration*0.6)
        self.assertTrue(iface_dac.complete())
        self.assertEqual(np.concatenate(self.samples).shape,
                         self.expected_size)

    def test_token_queue(self):
        channel = ni.DAQmxChannel(
            token=blocks.Tone(level=0),
            calibration=calibration.InterpCalibration.as_attenuation())
        iface_dac = ni.QueuedDAQmxPlayer(fs=self.fs, duration=self.duration)
        iface_dac.add_channel(channel)
        iface_dac.queue_init('FIFO')
        iface_dac.queue_append(self.repetitions, self.iti)
        iface_dac.play_queue()
        time.sleep(self.total_duration*1.1)
        self.assertTrue(iface_dac.complete())

        self.assertEqual(np.concatenate(self.samples).shape,
                         self.expected_size)


class TestDAQmxAttenControl(unittest.TestCase):

    def test_bitword(self):
        test_values = [
            ((254, 8, 'big-endian'), [1, 1, 1, 1, 1, 1, 1, 0]),
            ((254, 8, 'little-endian'), [0, 1, 1, 1, 1, 1, 1, 1]),
            ((128, 8, 'little-endian'), [0, 0, 0, 0, 0, 0, 0, 1]),
            ((128, 8, 'big-endian'), [1, 0, 0, 0, 0, 0, 0, 0]),
            ((128, 16, 'big-endian'), [0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 0, 0, 0, 0, 0, 0, 0]),
            ((128, 16, 'big-endian'), [0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 0, 0, 0, 0, 0, 0, 0]),
            ((254 | (128 << 8), 16, 'big-endian'), [1, 0, 0, 0, 0, 0, 0, 0,
                                                    1, 1, 1, 1, 1, 1, 1, 0]),
            ((254 | (128 << 8), 16, 'little-endian'), [0, 1, 1, 1, 1, 1, 1, 1,
                                                       0, 0, 0, 0, 0, 0, 0, 1]),
        ]
        for args, expected in test_values:
            actual = get_bits(*args)
            self.assertEqual(actual, expected)

    def test_gain_to_bits(self):
        atten = DAQmxAttenControl()
        self.assertEqual(atten._gain_to_byte(31.5), 255)
        self.assertEqual(atten._gain_to_byte(-95.5), 1)

        # Right, left
        actual = atten._gain_to_bits(31.0, -32.0)
        expected = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
