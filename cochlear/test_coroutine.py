import unittest
import collections

import numpy as np

from experiment.coroutine import coroutine, call
from cochlear.acquire import extract_epochs


class TestCoroutine(unittest.TestCase):

    def test_coroutine(self):
        epochs = []
        def accumulate(epoch):
            epochs.append(epoch)

        offsets = collections.deque()
        pipeline = extract_epochs(10, offsets, 75, call(accumulate))

        offsets.append(15)
        data = np.arange(100)
        pipeline.send(data[:10])
        self.assertEqual(0, len(epochs))
        pipeline.send(data[10:20])
        self.assertEqual(0, len(epochs))
        pipeline.send(data[20:30])
        self.assertEqual(1, len(epochs))
        offsets.append(20)
        offsets.append(30)
        offsets.append(80)
        self.assertEqual(1, len(epochs))
        pipeline.send(data[30:90])
        self.assertEqual(4, len(epochs))

        np.testing.assert_array_equal(data[15:25], epochs[0])
        np.testing.assert_array_equal(data[20:30], epochs[1])
        np.testing.assert_array_equal(data[30:40], epochs[2])


if __name__ == '__main__':
    unittest.main()
