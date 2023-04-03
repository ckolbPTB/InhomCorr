import numpy as np
import sys
sys.path.append('../ptbpyrecon/')

def normalise_image(im):
    """ Normalise image array

    Args:
        im (numpy array): Image input
        
    Returns: 
        (numpy array) : Normalised image 
    """
    assert isinstance(im, np.ndarray), 'Image should be numpy array. Got {}'.format(type(im))
    return((im - np.min(im))/(np.max(im) - np.min(im)))

    