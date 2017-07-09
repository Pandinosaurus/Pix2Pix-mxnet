import numpy as np
from util import lab_to_rgb_numpy
from util import rgb_to_lab_numpy


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
    # lab = rgb_to_lab_numpy(np.array(test_image))
    # np.testing.assert_array_almost_equal(lab, expected_lab, 4)    sess = tf.InteractiveSession()
    lab = rgb_to_lab_numpy(np.array(test_image))
    np.testing.assert_array_almost_equal(lab, expected_lab, 3)

def test_lab_to_rgb_np():
    # lab = rgb_to_lab_numpy(np.array(test_image))
    # np.testing.assert_array_almost_equal(lab, expected_lab, 4)    sess = tf.InteractiveSession()
    lab = rgb_to_lab_numpy(np.array(test_image))
    rgb = lab_to_rgb_numpy(lab)
    rgb = np.multiply(rgb, 256)
    rgb = np.ndarray.astype(rgb, np.uint8)
    np.testing.assert_array_almost_equal(rgb, test_image, 3)




