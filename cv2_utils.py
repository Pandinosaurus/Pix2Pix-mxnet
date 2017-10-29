import cv2

try:
    import mxnet as mx
    import numpy as np
except ImportError:
    pass


def show_mxnet_to_numpy_array(image_name, image_array):
    if mx and np \
            and isinstance(image_array, mx.ndarray.ndarray.NDArray) \
            and not isinstance(image_array, np.ndarray):
        image_array = image_array.asnumpy()
        __display(image_name, image_array)


def show_numpy_array(image_name, image_array):
    __display(image_name, image_array)


def __display(image_name, image_array):
    if (image_array.shape[-1] == 3):
        cv2.imshow(image_name, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        # cv2.imshow(image_name, image_array)
    else:
        cv2.imshow(image_name, image_array)
