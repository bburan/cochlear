import unittest
from cochlear import nidaqmx as ni
from neurogen import block_definitions as blocks
from neurogen import calibration
import PyDAQmx as _ni
import numpy as np
import time


import ctypes

class TestCoroutine(unittest.TestCase):

    class Accumulator(object):

        def __init__(self):
            self.results = []

        def send(self, data):
            self.results.append(data)

    def setUp(self):
        self.epoch_size = 10
        self.target = self.Accumulator()
        self.coroutine = ni.triggered_ai_from_continuous(self.epoch_size,
                                                         self.target)

    def test_coroutine(self):
        # First send
        indices = 5, 23, 40, 55
        analog = np.arange(200)
        digital = np.zeros_like(analog)
        expected = []
        for i in indices:
            digital[i] = 1
            expected.append(analog[i:i+self.epoch_size])
        self.coroutine.send((digital, analog))
        np.testing.assert_array_equal(self.target.results, expected)

        # Second send
        indices = 0, 11, 22, 50
        analog = np.random.random(200)
        digital = np.zeros_like(analog)
        for i in indices:
            digital[i] = 1
            expected.append(analog[i:i+self.epoch_size])
        self.coroutine.send((digital, analog))
        np.testing.assert_array_equal(self.target.results, expected)

        # Third send
        analog = np.random.random(10)
        digital = np.zeros_like(analog)
        digital[0] = 1
        expected.append(analog[:self.epoch_size])
        self.coroutine.send((digital, analog))
        np.testing.assert_array_equal(self.target.results, expected)

        # Third send
        analog = np.random.random(10)
        digital = np.zeros_like(analog)
        digital[0] = 1
        expected.append(analog[:self.epoch_size])
        self.coroutine.send((digital, analog))
        np.testing.assert_array_equal(self.target.results, expected)
        print len(self.target.results)
        print len(expected)

        # Fourth send
        analog = np.random.random(50)
        digital = np.zeros_like(analog)
        digital = np.zeros_like(analog)
        indices = 11, 22, 45, 49
        for i in indices:
            digital[i] = 1
            if i < 40:
                expected.append(analog[i:i+self.epoch_size])
        self.coroutine.send((digital, analog))
        np.testing.assert_array_equal(self.target.results, expected)


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
            actual = ni.get_bits(*args)
            self.assertEqual(actual, expected)

    def test_gain_to_bits(self):
        atten = ni.DAQmxAttenControl()
        self.assertEqual(atten._gain_to_byte(31.5), 255)
        self.assertEqual(atten._gain_to_byte(-95.5), 1)

        # Right, left
        actual = atten._gain_to_bits(31.0, -32.0)
        expected = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
