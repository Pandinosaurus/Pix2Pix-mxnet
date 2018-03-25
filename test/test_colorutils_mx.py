import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from util import lab_to_rgb
from util import rgb_to_lab


test_image = [
    [[1, 255, 125], [1, 255, 125], [1, 255, 125]],
    [[1, 255, 125], [1, 255, 125], [1, 255, 125]],
    [[1, 255, 125], [1, 255, 125], [1, 255, 125]]
]

expected_lab = [[[88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776]],

            [[88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776]],

            [[88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776],
             [88.448891, -77.174858,  47.935776]]]


def test_rgb_to_lab_np():
    cpu_context = mx.cpu()
    lab = rgb_to_lab(nd.array(test_image), ctx=cpu_context)
    np.testing.assert_array_almost_equal(lab.asnumpy(), expected_lab, 3)


def test_lab_to_rgb_np():
    cpu_context = mx.cpu()
    lab = rgb_to_lab(nd.array(test_image), ctx=cpu_context)
    rgb = lab_to_rgb(lab, ctx=cpu_context)
    rgb = nd.multiply(rgb, nd.array([256]))
    rgb = nd.cast(rgb, np.uint8)
    np.testing.assert_array_almost_equal(rgb.asnumpy(), test_image, 3)




