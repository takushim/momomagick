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

def paste_slices (src_shape, tgt_shape, center = False):
    if center:
        shifts = (np.array(tgt_shape) - np.array(src_shape)) // 2
    else:
        shifts = np.array([0 for i in range(len(src_shape))])

    src_starts = [min(-x, y) if x < 0 else 0 for x, y in zip(shifts, src_shape)]
    src_bounds = np.minimum(src_shape, tgt_shape - shifts)
    slices_src = tuple([slice(x, y) for x, y in zip(src_starts, src_bounds)])

    tgt_starts = [0 if x < 0 else min(x, y) for x, y in zip(shifts, tgt_shape)]
    tgt_bounds = np.minimum(src_shape + shifts, tgt_shape)
    slices_tgt = tuple([slice(x, y) for x, y in zip(tgt_starts, tgt_bounds)])

    return [slices_src, slices_tgt]

def resize (image_array, shape, center = False):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = paste_slices(image_array.shape, shape, center)
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

def z_rotate (input_image, angle = 0.0, gpu_id = None):
    rotate_tuple = (0, 2)
    if gpu_id is None:
        image = ndimage.rotate(input_image, angle, axes = rotate_tuple, reshape = False)
    else:
        image = cp.asarray(input_image)
        image = cpimage.rotate(image, angle, axes = rotate_tuple, order = 1, reshape = False)
        image = cp.asnumpy(image)

    return image

def affine_transform (input_image, matrix, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.affine_transform(input_image, matrix)
    else:
        output_image = cpimage.affine_transform(cp.array(input_image), cp.array(matrix))
        output_image = cp.asnumpy(output_image)
    return output_image

