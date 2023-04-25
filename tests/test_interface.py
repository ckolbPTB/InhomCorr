import pytest
import torch

import inhomcorr.interfaces.mr_data_interface as interface


@pytest.fixture
def numPixels():
    return 16


@pytest.fixture
def image2DSize(numPixels):
    return (1, 1, numPixels, numPixels)


@pytest.fixture
def data2D(image2DSize):
    return torch.rand(image2DSize)


@pytest.fixture
def image2D(data2D):
    img = interface.ImageData()
    img.data = data2D
    return img


def test_mr_image_shape(image2D, image2DSize):
    assert image2D.shape == image2DSize
