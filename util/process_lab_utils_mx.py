import mxnet.ndarray as nd


def preprocess_lab(lab):
    l_chan, a_chan, b_chan = nd.split(lab, axis=2, num_outputs=3)
    l_chan = nd.divide(l_chan, 50) - 1
    a_chan = nd.divide(a_chan, 110)
    b_chan = nd.divide(b_chan, 110)
    return [l_chan, a_chan, b_chan]