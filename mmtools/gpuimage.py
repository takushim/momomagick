#!/usr/bin/env python

import sys
import numpy as np
from scipy import ndimage
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    if gpu_id is None:
        print("GPU ID not specified. Continuing with CPU.")
        return None
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)
    return device

def paste_slices (src_shape, tgt_shape, center = False, shifts = None):
    if shifts is None:
        if center:
            shifts = (np.array(tgt_shape) - np.array(src_shape)) // 2
        else:
            shifts = np.array([0 for i in range(len(src_shape))])
    else:
        shifts = np.array(shifts)

    src_starts = [min(-x, y) if x < 0 else 0 for x, y in zip(shifts, src_shape)]
    src_bounds = np.minimum(src_shape, tgt_shape - shifts)
    slices_src = tuple([slice(x, y) for x, y in zip(src_starts, src_bounds)])

    tgt_starts = [0 if x < 0 else min(x, y) for x, y in zip(shifts, tgt_shape)]
    tgt_bounds = np.minimum(src_shape + shifts, tgt_shape)
    slices_tgt = tuple([slice(x, y) for x, y in zip(tgt_starts, tgt_bounds)])

    return [slices_src, slices_tgt]

def resize (image_array, shape, center = False):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = paste_slices(image_array.shape, shape, center = center)
    resized_array[slices_tgt] = image_array[slices_src].copy()
    return resized_array

def crop (image_array, origin, shape):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = paste_slices(image_array.shape, shape, shifts = -origin)
    resized_array[slices_tgt] = image_array[slices_src].copy()
    return resized_array

def z_zoom (input_image, ratio = 1.0, gpu_id = None):
    if len(input_image.shape) < 3 or input_image.shape[0] == 1:
        print("Cannot zoom 2D images.")
        return input_image

    if gpu_id is None:
        output_image = ndimage.zoom(input_image, (ratio, 1.0, 1.0))
    else:
        output_image = cpimage.zoom(cp.array(input_image), (ratio, 1.0, 1.0))
        output_image = cp.asnumpy(output_image)
    return output_image

def rotate (input_image, angle = 0.0, axis = 0, gpu_id = None):
    if axis == 0 or axis == 'z' or axis == 'Z':
        rotate_tuple = (1, 2)
    elif axis == 1 or axis == 'y' or axis == 'Y':
        rotate_tuple = (0, 2)
    elif axis == 2 or axis == 'x' or axis == 'X':
        rotate_tuple = (0, 1)
    else:
        raise Exception('Invalid axis was specified.')

    if gpu_id is None:
        image = ndimage.rotate(input_image, angle, axes = rotate_tuple, reshape = False)
    else:
        image = cp.asarray(input_image)
        image = cpimage.rotate(image, angle, axes = rotate_tuple, order = 1, reshape = False)
        image = cp.asnumpy(image)

    return image

def affine_transform (input_image, matrix, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.affine_transform(input_image, matrix, mode = 'grid-constant')
    else:
        output_image = cpimage.affine_transform(cp.array(input_image), cp.array(matrix), mode = 'grid-constant')
        output_image = cp.asnumpy(output_image)
    return output_image

def shift (input_image, shifts, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.interpolation.shift(input_image, shifts)
    else:
        output_image = cpimage.interpolation.shift(cp.array(input_image), shifts)
        output_image = cp.asnumpy(output_image)
    return output_image
