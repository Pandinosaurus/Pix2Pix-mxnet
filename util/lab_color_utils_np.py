import numpy as np
import cv2

# Converted to numpy by skirdye
# Original from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

def rgb_to_lab_numpy(image_srgb):
    srgb = _check_image(image_srgb)
    if image_srgb.max() > 1:
        srgb = _normalize_rgb_image(srgb)

    srgb_pixels = np.reshape(srgb, [-1, 3])

    linear_mask = srgb_pixels <= 0.04045
    exponential_mask = srgb_pixels > 0.04045

    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = np.matrix([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ])
    xyz_pixels = np.matmul(rgb_pixels, rgb_to_xyz)  # OK
    xyz_normalized_pixels = np.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])
    epsilon = 6 / 29
    linear_mask = xyz_normalized_pixels <= (epsilon ** 3)
    exponential_mask = xyz_normalized_pixels > (epsilon ** 3)
    apply_linear_mask = np.multiply((xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29), linear_mask)
    apply_exp_mask = np.multiply(np.power(xyz_normalized_pixels, (1 / 3)), exponential_mask)
    fxfyfz_pixels = apply_linear_mask + apply_exp_mask

    fxfyfz_to_lab = np.matrix([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ])

    lab_pixels = np.matmul(fxfyfz_pixels, fxfyfz_to_lab) + np.matrix([-16.0, 0.0, 0.0])

    return np.ravel(lab_pixels).reshape(srgb.shape)


def lab_to_rgb_numpy(lab):
    lab = _check_image(lab)
    lab_pixels = np.reshape(lab, [-1, 3])
    lab_to_fxfyfz = np.array([
            #   fx      fy        fz
            [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
            [1 / 500.0, 0.0, 0.0],  # a
            [0.0, 0.0, -1 / 200.0],  # b
        ])
    fxfyfz_pixels = np.dot(lab_pixels + np.array([16.0, 0.0, 0.0]), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6 / 29
    linear_mask = fxfyfz_pixels <= epsilon
    exponential_mask = fxfyfz_pixels > epsilon


    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

    xyz_pixels = np.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])
    xyz_to_rgb = np.matrix([
            #     r           g          b
            [3.2404542, -0.9692660, 0.0556434],  # x
            [-1.5371385, 1.8760108, -0.2040259],  # y
            [-0.4985314, 0.0415560, 1.0572252],  # z
    ])
    rgb_pixels = np.dot(xyz_pixels, xyz_to_rgb)
    np.clip(rgb_pixels, 0.0, 1.0, out=rgb_pixels)


    linear_mask = rgb_pixels <= 0.0031308
    exponential_mask = rgb_pixels > 0.0031308

    step1 = np.multiply(np.multiply(rgb_pixels, 12.92), linear_mask)
    step2 = np.multiply(np.multiply(np.power(rgb_pixels, (1 / 2.4)), 1.055) - 0.055, exponential_mask)
    srgb_pixels = step1 + step2

    return np.ravel(srgb_pixels).reshape(lab.shape)

def _denormalize_rgb_image(image_array):
    if(isinstance(image_array, np.ndarray)):
        return __denormalize_rgb_image_np(image_array)
    return image_array

def _normalize_rgb_image(image_array):
    if(isinstance(image_array, np.ndarray)):
        return __normalize_rgb_image_np(image_array)


def __normalize_rgb_image_np(arr):
    # type: (np.ndarray) -> np.ndarray
    """
    :param arr: (N, N, 3L) matrix representing image of size N*N
    :return: Normalized array of with RGB values ranging from 0 to 1
    """
    return np.ndarray.astype(np.divide(arr, 255.0), np.float32)

def __denormalize_rgb_image_np(arr):
    # type: (np.ndarray) -> np.ndarray
    """
    :param arr: (N, N, 3L) matrix representing image of size N*N
    :return: Normalized array of with RGB values ranging from 0 to 1
    """
    np.multiply(arr, 255.0)

def _check_image(image):
    assert image.shape[-1] == 3
    if len(image.shape) not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")
    shape = list(image.shape)
    shape[-1] = 3
    return np.reshape(image, shape)
