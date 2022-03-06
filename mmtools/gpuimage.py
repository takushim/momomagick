#!/usr/bin/env python

import numpy as np
from scipy import ndimage
from logging import getLogger
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

logger = getLogger(__name__)

def add_gpu_argument (parser, gpu_id = None):
    parser.add_argument('-g', '--gpu-id', default = gpu_id, help='GPU ID')

def parse_gpu_argument (args):
    turn_on_gpu(args.gpu_id)
    return args.gpu_id

def turn_on_gpu (gpu_id):
    device = cp.cuda.Device(gpu_id)
    device.use()
    logger.info("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    logger.info("Free memory: {0}".format(device.mem_info))
    return device

def pasting_slices (src_shape, tgt_shape, centering = False, offset = None):
    if centering:
        shifts = (np.array(tgt_shape) - np.array(src_shape)) // 2
    else:
        shifts = np.array([0] * len(src_shape))

    if offset is not None:
        shifts = shifts + np.array(offset)

    src_starts = [min(-x, y) if x < 0 else 0 for x, y in zip(shifts, src_shape)]
    src_bounds = np.minimum(src_shape, tgt_shape - shifts)
    slices_src = tuple([slice(x, y) for x, y in zip(src_starts, src_bounds)])

    tgt_starts = [0 if x < 0 else min(x, y) for x, y in zip(shifts, tgt_shape)]
    tgt_bounds = np.minimum(src_shape + shifts, tgt_shape)
    slices_tgt = tuple([slice(x, y) for x, y in zip(tgt_starts, tgt_bounds)])

    return [slices_src, slices_tgt]

def resize (image_array, shape, centering = False):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = pasting_slices(image_array.shape, shape, centering = centering)
    resized_array[slices_tgt] = image_array[slices_src].copy()
    return resized_array

def crop (image_array, origin, shape):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = pasting_slices(image_array.shape, shape, offset = -origin)
    resized_array[slices_tgt] = image_array[slices_src].copy()
    return resized_array

def zoom (input_image, ratio = 1.0, gpu_id = None):
    logger.warning("This function will be deplicated.")
    if isinstance(ratio, (list, tuple, np.ndarray)):
        if len(input_image.shape) > len(ratio):
            ratio = [1.0 for i in range(len(input_image.shape) - len(ratio))] + ratio
        elif len(input_image.shape) < len(ratio):
            ratio = ratio[(len(ratio) - len(input_image.shape)):len(ratio)]
    else:
        ratio = [ratio for i in input_image.shape]

    if np.allclose(ratio, 1.0):
        return input_image.copy()

    if gpu_id is None:
        output_image = ndimage.zoom(input_image, ratio)
    else:
        output_image = cpimage.zoom(cp.array(input_image), ratio)
        output_image = cp.asnumpy(output_image)

    return output_image

def z_zoom (input_image, ratio = 1.0, gpu_id = None):
    logger.warning("This function will be deplicated.")
    if len(input_image.shape) < 3 or input_image.shape[0] == 1:
        logger.info("Skipping z-zooming of a 2D image.")
        output_image = input_image.copy()
    else:
        output_image = zoom(input_image, ratio = (ratio, 1.0, 1.0), gpu_id = gpu_id)
    
    return output_image

def scale (input_image, ratio, gpu_id = None):
    if np.allclose(ratio, 1.0) == False:
        if gpu_id is None:
            output_image = ndimage.zoom(input_image, ratio)
        else:
            output_image = cpimage.zoom(cp.array(input_image), ratio)
            output_image = cp.asnumpy(output_image)
    else:
        output_image = input_image.copy()

    return output_image

def expand_ratio (ratio):
    if isinstance(ratio, (list, tuple, np.ndarray)):
        if len(ratio) == 0:
            ratio = [1.0, 1.0, 1.0]
        elif len(ratio) == 1:
            ratio = [ratio[0], ratio[0], ratio[0]]
        elif len(ratio) == 2:
            ratio = [ratio[0], ratio[1], ratio[1]]
    else:
        ratio = [ratio, ratio, ratio]

    return ratio

def rotate (input_image, angle, rot_tuple, gpu_id = None):
    if gpu_id is None:
        image = ndimage.rotate(input_image, angle, axes = rot_tuple, reshape = False)
    else:
        image = cp.asarray(input_image)
        image = cpimage.rotate(image, angle, axes = rot_tuple, order = 1, reshape = False)
        image = cp.asnumpy(image)

    return image

def rotate_by_axis (input_image, angle, axis, gpu_id = None):
    rot_tuple = axis_to_tuple(axis)
    return rotate(input_image, angle, rot_tuple, gpu_id = gpu_id)

def axis_to_tuple (axis):
    if axis == 0 or axis == 'z' or axis == 'Z':
        rotate_tuple = (1, 2)
    elif axis == 1 or axis == 'y' or axis == 'Y':
        rotate_tuple = (0, 2)
    elif axis == 2 or axis == 'x' or axis == 'X':
        rotate_tuple = (0, 1)
    else:
        raise Exception('Invalid axis was specified.')

    return rotate_tuple

def affine_transform (input_image, matrix, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.affine_transform(input_image, matrix, mode = 'grid-constant')
    else:
        output_image = cpimage.affine_transform(cp.array(input_image), cp.array(matrix), mode = 'grid-constant')
        output_image = cp.asnumpy(output_image)
    return output_image

def shift (input_image, offset, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.interpolation.shift(input_image, offset)
    else:
        output_image = cpimage.interpolation.shift(cp.array(input_image), offset)
        output_image = cp.asnumpy(output_image)
    return output_image
