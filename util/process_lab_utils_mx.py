import mxnet.ndarray as nd
import numpy as np

# Certain useful are still work in progress in the mxnet.ndarray api
# such as np.squeeze
# Fall back to numpy

def preprocess_lab(lab):
    """
    Normalize lab colorspace to a format friendly for training via neural nets
    We move to a [-1, 1] numerical space
    :param lab: denormalized matrix representing image in LAB color space
    :return: normalized matrix representing image in LAB color space
    """
    l_chan, a_chan, b_chan = nd.split(lab, axis=2, num_outputs=3)
    l_chan = nd.divide(l_chan, 50) - 1
    a_chan = nd.divide(a_chan, 110)
    b_chan = nd.divide(b_chan, 110)
    return [l_chan, a_chan, b_chan]


def deprocess_lab(lightness_chan, a_chan, b_chan):
    """
    :param lightness_chan: Lightness of an image, usually 1xNxN size matrix
    :param a_chan: R^(1xNxN) matrix, representing a colors
    :param b_chan: R^(1xNxN)  matrix, representing b colors
    :return: a R^(3xNxN) matrix representing image in lab space that can be converted to RGB
    """
    return nd.stack((lightness_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110, axis=3)