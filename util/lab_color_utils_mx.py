import mxnet as mx
import mxnet.ndarray as nd
import numpy as np


# Converted to MXNet by skirdey
# Original from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

def rgb_to_lab(image_srgb, ctx=None):

    if ctx is None:
        raise ValueError("ctx can not be None")

    if image_srgb is None:
        raise ValueError("image_srgb can not be None")

    with mx.Context(ctx):

        srgb = __check_image(image_srgb)

        if nd.max(srgb).asscalar() > 1:
            srgb = __normalize_rgb_image(srgb)

        srgb_pixels = nd.reshape(srgb, [-1, 3])

        linear_mask = nd.cast(srgb_pixels <= 0.04045, dtype='float32')
        exponential_mask = nd.cast(srgb_pixels > 0.04045, dtype='float32')
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_to_xyz = nd.array([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ])
        xyz_pixels = nd.linalg_gemm2(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)
        # normalize for D65 white point
        xyz_normalized_pixels = nd.multiply(xyz_pixels, nd.array([1 / 0.950456, 1.0, 1 / 1.088754]))

        epsilon = 6 / 29
        linear_mask = nd.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype='float32')
        exponential_mask = nd.cast(xyz_normalized_pixels > (epsilon ** 3), dtype='float32')
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                                                                                                  xyz_normalized_pixels ** (
                                                                                                  1 / 3)) * exponential_mask
            # convert to lab
        fxfyfz_to_lab = nd.array([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
        lab_pixels = nd.linalg_gemm2(fxfyfz_pixels, fxfyfz_to_lab) + nd.array([-16.0, 0.0, 0.0])

        return nd.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab, ctx=None):
    if ctx is None:
        raise ValueError("ctx can not be None")

    if lab is None:
        raise ValueError("lab can not be None")

    with mx.Context(ctx):
        lab = __check_image(lab)
        lab_pixels = lab.reshape([-1, 3])
        lab_to_fxfyfz = nd.array([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ], ctx=ctx)
        fxfyfz_pixels = nd.dot(lab_pixels + nd.array([16.0, 0.0, 0.0], ctx=ctx), lab_to_fxfyfz)

        # convert to xyz
        epsilon = 6 / 29
        linear_mask = fxfyfz_pixels <= epsilon
        exponential_mask = fxfyfz_pixels > epsilon

        xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

        xyz_pixels = nd.multiply(xyz_pixels, nd.array([0.950456, 1.0, 1.088754]))
        xyz_to_rgb =nd.array([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
        ])
        rgb_pixels = nd.dot(xyz_pixels, xyz_to_rgb)
        nd.clip(rgb_pixels, 0.0, 1.0, out=rgb_pixels)

        linear_mask = rgb_pixels <= 0.0031308
        exponential_mask = rgb_pixels > 0.0031308

        step1 = nd.multiply(nd.multiply(rgb_pixels, 12.92), linear_mask)
        step2 = nd.multiply(nd.multiply(nd.power(rgb_pixels, (1 / 2.4)), 1.055) - 0.055, exponential_mask)
        srgb_pixels = step1 + step2

        return srgb_pixels.reshape(lab.shape)


def __normalize_rgb_image(arr):
    return nd.cast(arr, "float32") / 255.0


def __check_image(image):
    assert image.shape[-1] == 3
    if len(image.shape) not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")
    shape = list(image.shape)
    shape[-1] = 3
    return nd.reshape(image, shape)


