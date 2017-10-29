import mxnet.ndarray as nd

# Certain useful are still work in progress in the mxnet.ndarray api
# such as np.squeeze
# Fall back to numpy

def preprocess_lab(lab):
    """
    Normalize lab color space to a format friendly for training via neural nets
    We move to a [-1, 1] numerical space
    :param lab: de-normalized matrix representing image in LAB color space
    :return: normalized matrix representing image in LAB color space
    """
    l_chan, a_chan, b_chan = nd.split(lab, axis=2, num_outputs=3)
    l_chan = nd.divide(l_chan, 50) - 1
    a_chan = nd.divide(a_chan, 110)
    b_chan = nd.divide(b_chan, 110)
    return [l_chan, a_chan, b_chan]