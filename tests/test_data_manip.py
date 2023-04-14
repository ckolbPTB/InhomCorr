import unittest

import numpy as np

from inhomcorr.data_manip import normalise_image


class TestDataManip(unittest.TestCase):

    def test_normalise_image(self):
        im = np.random.randn(20, 20)
        im_norm = normalise_image(im)
        self.assertEqual(np.min(im_norm), 0)
        self.assertEqual(np.max(im_norm), 1)
