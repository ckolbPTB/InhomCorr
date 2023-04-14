import unittest

import numpy as np

from inhomcorr.dummyclass import DummyClass


class TestDummyClass(unittest.TestCase):

    def test_dummy_class(self):
        dummy = DummyClass(np.random.randn(20, 20))
        self.assertEqual(dummy.id, 0)
        dummy.increase_id()
        self.assertEqual(dummy.id, 1)
