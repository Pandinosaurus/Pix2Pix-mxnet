import cv2
import mxnet as mx
import numpy as np


def visualize_cv2(label, img_arr):
    cv2.imshow(label, cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


def visualize_rgb(label, img_arr):
    cv2.imshow(label, img_arr)
    cv2.waitKey(1)


def visualize_ndarray(label, nd_array, color_space=None):
    np_array = np.transpose(
        np.squeeze(
            nd_array.asnumpy(),
            0
        ),
        (1, 2, 0)
    )

    if color_space:
        np_array = cv2.cvtColor(np_array, color_space)

    cv2.imshow(label, np_array)
    cv2.waitKey(1)
