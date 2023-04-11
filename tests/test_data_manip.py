import numpy as np

from inhomcorr import data_manip


def test_normalise_image():
    im = np.random.randn(20, 20)
    im_norm = data_manip.normalise_image(im)
    assert np.min(im_norm) == 0
    assert np.max(im_norm) == 1
