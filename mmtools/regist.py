#!/usr/bin/env python

import sys, argparse
import numpy as np
from functools import reduce
from scipy import ndimage, optimize
from mmtools import mmtiff
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)
    return device

def window_mat (shape, window_func):
    if window_func is None:
        matrix = np.ones(shape)
    else:
        hanning_list = [np.hanning(x) for x in shape]
        mesh_list = np.meshgrid(*hanning_list, indexing = 'ij')
        matrix = reduce(lambda x, y: x * y, mesh_list)
    return matrix

def normalize (input_image, p_min = 1, p_max = 99):
    clip_min = np.percentile(input_image, p_min)
    clip_max = np.percentile(input_image, p_max)
    return (input_image.clip(clip_min, clip_max).astype(float) - clip_min) / (clip_max - clip_min)

def csv_to_matrix (csv_text):
    params = [float(x) for x in csv_text.split(',')]
    if len(params) == 2:
        matrix = drift_to_matrix_2d(params)
    elif len(params) == 3:
        matrix = drift_to_matrix_2d(params)
    elif len(params) == 6:
        matrix = params_to_matrix_2d(params)
    elif len(params) == 9:
        matrix = np.array(params).reshape(3, 3)
    elif len(params) == 12:
        matrix = params_to_matrix_3d(params)
    elif len(params) == 16:
        matrix = np.array(params).reshape(4, 4)
    else:
        print("Cannot make a matrix.", params)
        matrix = np.array([1.0])
    return matrix

def params_to_matrix_2d (params):
    return np.array([params[0:3], params[3:6], [0.0, 0.0, 1.0]])

def params_to_matrix_3d (params):
    return np.array([params[0:4], params[4:8], params[8:12], [0.0, 0.0, 0.0, 1.0]])

def drift_to_matrix_2d (params):
    return np.array([[1.0, 0.0, params[0]], [0.0, 1.0, params[1]], [0.0, 0.0, 1.0]])

def drift_to_matrix_3d (params):
    return np.array([[1.0, 0.0, 0.0, params[0]], [0.0, 1.0, 0.0, params[1]], \
                     [0.0, 0.0, 1.0, params[2]], [0.0, 0.0, 0.0, 1.0]])

def affine_transform (input_image, matrix, gpu_id = None):
    if gpu_id is None:
        output_image = ndimage.affine_transform(input_image, matrix)
    else:
        output_image = cpimage.affine_transform(cp.array(input_image), cp.array(matrix))
        output_image = cp.asnumpy(output_image)
    return output_image


class Poc:
    def __init__ (self, ref_image, window_func = np.hanning, gpu_id = None):
        self.window_mat = window_mat(ref_image.shape, window_func)
        self.center = np.array(ref_image.shape) // 2
        self.gpu_id = gpu_id
        if gpu_id is None:
            self.ref_fft_conj = np.conj(np.fft.fftn(ref_image * self.hanning_mat))
        else:
            self.window_mat = cp.array(self.window_mat)
            self.ref_fft_conj = cp.conj(cp.fft.fftn(cp.array(ref_image) * self.window_mat))

    def poc_image(self, input_image):
        if self.gpu_id is None:
            image_fft = np.fft.fftn(input_image * self.window_mat)
            corr_image = self.ref_fft_conj * image_fft / np.abs(self.ref_fft_conj * image_fft)
            poc_image = np.fft.fftshift(np.real(np.fft.ifftn(corr_image)))
        else:
            image = cp.array(input_image)
            image_fft = cp.fft.fftn(image * self.window_mat)
            corr_image = self.ref_fft_conj * image_fft / cp.abs(self.ref_fft_conj * image_fft)
            poc_image = cp.fft.fftshift(cp.real(cp.fft.ifftn(corr_image)))
            poc_image = cp.asnumpy(poc_image)
        return poc_image

    def regist(self, input_image):
        poc_image = self.poc_image(input_image)
        max_pos = ndimage.maximum_position(poc_image)
        max_val = poc_image[max_pos]
        shift = (max_pos - self.center)
        return {'shift': shift, 'corr': max_val}

class Affine:
    def __init__ (self, ref_image, window_func = np.hanning, gpu_id = None):
        self.window_mat = window_mat(ref_image.shape, window_func)
        self.ref_float = normalize(ref_image)
        self.gpu_id = gpu_id
        if gpu_id is not None:
            self.window_mat = cp.array(self.window_mat)
            self.ref_float = cp.array(self.ref_float)

    def regist (self, input_image, init_shift, optimizing_method = "Powell", transport_only = False):
        if len(input_image.shape) == 2:
            init_params = np.array([1.0, 0.0, init_shift[0], 0.0, 1.0, init_shift[1]])
            if transport_only:
                params_to_matrix = drift_to_matrix_2d
            else:
                params_to_matrix = params_to_matrix_2d
        elif len(input_image.shape) == 3:
            init_params = np.array([1.0, 0.0, 0.0, init_shift[0], 0.0, 1.0, 0.0, init_shift[1], 0.0, 0.0, 1.0, init_shift[2]])
            if transport_only:
                params_to_matrix = drift_to_matrix_3d
            else:
                params_to_matrix = params_to_matrix_3d
        else:
            raise Exception("Input images must be 2D or 3D grayscale.")

        if self.gpu_id is None:
            image_float = normalize(input_image)
            def error_func (params):
                matrix = params_to_matrix(params)
                trans_float = ndimage.affine_transform(image_float, matrix)
                error = np.sum((self.ref_float - trans_float) * (self.ref_float - trans_float) * self.window_mat)
                return error
        else:
            image_float = cp.array(normalize(input_image))
            def error_func (params):
                matrix = cp.array(params_to_matrix(params))
                trans_float = cpimage.affine_transform(image_float, matrix)
                error = cp.asnumpy(cp.sum((self.ref_float - trans_float) * (self.ref_float - trans_float) * self.window_mat))
                return error

        results = optimize.minimize(error_func, init_params, method = optimizing_method)
        return {'matrix': params_to_matrix(results.x), 'results': results}

