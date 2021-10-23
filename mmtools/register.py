#!/usr/bin/env python

import sys, argparse
import numpy as np
from functools import reduce
from transforms3d import affines, euler
from scipy import ndimage, optimize
from . import gpuimage
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

optimizing_methods = ["Auto", "Powell", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B", "SLSQP", "None"]
registering_methods = ["Full", "Rigid", "Drift", "None"]

def turn_on_gpu (gpu_id):
    return gpuimage.turn_on_gpu(gpu_id)

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

def params_to_matrix_2d (params):
    return np.array([params[0:3], params[3:6], [0.0, 0.0, 1.0]])

def params_to_matrix_3d (params):
    return np.array([params[0:4], params[4:8], params[8:12], [0.0, 0.0, 0.0, 1.0]])

def drift_to_matrix_2d (params):
    return np.array([[1.0, 0.0, params[0]], [0.0, 1.0, params[1]], [0.0, 0.0, 1.0]])

def drift_to_matrix_3d (params):
    return np.array([[1.0, 0.0, 0.0, params[0]], [0.0, 1.0, 0.0, params[1]], \
                     [0.0, 0.0, 1.0, params[2]], [0.0, 0.0, 0.0, 1.0]])

def rbm_to_matrix_2d (params):
    return np.array([[np.cos(params[2]), -np.sin(params[2]), params[0]], \
                     [np.sin(params[2]),  np.cos(params[2]), params[1]], \
                     [0.0, 0.0, 1.0]])

def rbm_to_matrix_3d (params):
    matrix = np.array([[1.0, 0.0, 0.0, params[0]], [0.0, 1.0, 0.0, params[1]], \
                       [0.0, 0.0, 1.0, params[2]], [0.0, 0.0, 0.0, 1.0]])
    matrix[0:3, 0:3] = euler.euler2mat(*params[3:6])
    return matrix

def decompose_matrix (matrix):
    # interpret the affine matrix
    trans, rot_mat, zoom, shear = affines.decompose(matrix)
    if rot_mat.shape == (3, 3):
        rot_angles = np.array(euler.mat2euler(rot_mat)) / np.pi * 180
    elif rot_mat.shape == (2, 2):
        rot_angles = np.array([np.arccos(rot_mat[0, 0])]) / np.pi * 180
    else:
        print("Cannot analyze the roration matrix:", rot_mat)
        rot_angles = None

    return {'transport': trans, 'rotation_matrix': rot_mat, 'rotation_angles': rot_angles, \
            'zoom': zoom, 'shear': shear}
 
def register (ref_image, input_image, init_shift = None, gpu_id = None, reg_method = "Full", opt_method = "Powell"):
    if init_shift is None:
        # calculate POCs for pre-registration
        poc_register = Poc(ref_image, gpu_id = gpu_id)
        poc_result = poc_register.register(input_image)
        poc_register = None
        init_shift = poc_result['shift']

    # calculate an affine matrix for registration
    if len(input_image.shape) < 3 or input_image.shape[0] == 1:
        init_shift = init_shift[1:]
        affine_register = Affine(ref_image[0], gpu_id = gpu_id)
        affine_result = affine_register.register(input_image[0], init_shift, opt_method = opt_method, reg_method = reg_method)
    else:
        affine_register = Affine(ref_image, gpu_id = gpu_id)
        affine_result = affine_register.register(input_image, init_shift, opt_method = opt_method, reg_method = reg_method)

    return affine_result

class Poc:
    def __init__ (self, ref_image, window_func = np.hanning, gpu_id = None):
        self.window_mat = window_mat(ref_image.shape, window_func)
        self.center = np.array(ref_image.shape) // 2
        self.gpu_id = gpu_id
        if gpu_id is None:
            self.ref_fft_conj = np.conj(np.fft.fftn(ref_image * self.window_mat))
        else:
            self.window_mat = cp.array(self.window_mat)
            self.ref_fft_conj = cp.conj(cp.fft.fftn(cp.array(ref_image) * self.window_mat))

    def poc_image (self, input_image):
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

    def register (self, input_image):
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

    def register (self, input_image, init_shift, opt_method = "Powell", reg_method = "Full"):
        if len(input_image.shape) == 2:
            if reg_method == 'None':
                if init_shift is None:
                    init_params = [0.0, 0.0]
                else:
                    init_params = init_shift.flatten()
                params_to_matrix = drift_to_matrix_2d
            elif reg_method == 'Drift':
                init_params = init_shift.flatten()
                params_to_matrix = drift_to_matrix_2d
            elif reg_method == 'Rigid':
                init_params = np.array([init_shift[0], init_shift[1], 0.0])
                params_to_matrix = rbm_to_matrix_2d
            elif reg_method == 'Full':
                init_params = np.array([1.0, 0.0, init_shift[0], 0.0, 1.0, init_shift[1]])
                params_to_matrix = params_to_matrix_2d
            else:
                raise Exception("Unknown registration method:", reg_method)
        elif len(input_image.shape) == 3:
            if reg_method == 'None':
                if init_shift is None:
                    init_params = [0.0, 0.0, 0.0]
                else:
                    init_params = init_shift.flatten()
                params_to_matrix = drift_to_matrix_3d
            elif reg_method == 'Drift':
                init_params = init_shift.flatten()
                params_to_matrix = drift_to_matrix_3d
            elif reg_method == 'Rigid':
                init_params = np.array([init_shift, [0.0, 0.0, 0.0]]).flatten()
                params_to_matrix = rbm_to_matrix_3d
            elif reg_method == 'Full':
                init_params = np.array([1.0, 0.0, 0.0, init_shift[0], 0.0, 1.0, 0.0, init_shift[1], 0.0, 0.0, 1.0, init_shift[2]])
                params_to_matrix = params_to_matrix_3d
            else:
                raise Exception("Unknown registration method:", reg_method)
        else:
            raise Exception("Input images must be 2D or 3D grayscale.")

        if self.gpu_id is None:
            image_float = normalize(input_image)
            def error_func (params):
                matrix = params_to_matrix(params)
                trans_float = ndimage.affine_transform(image_float, matrix) * self.window_mat
                error = np.sum((self.ref_float - trans_float) * (self.ref_float - trans_float))
                return error
        else:
            image_float = cp.array(normalize(input_image))
            def error_func (params):
                matrix = cp.array(params_to_matrix(params))
                trans_float = cpimage.affine_transform(image_float, matrix) * self.window_mat
                error = cp.asnumpy(cp.sum((self.ref_float - trans_float) * (self.ref_float - trans_float)))
                return error

        if reg_method == "None":
            results = optimize.OptimizeResult()
            results.x = init_params
            results.success = True
            results.message = 'Registration not performed.'
            final_matrix = params_to_matrix(init_params)
        else:
            if opt_method == "None":
                results = optimize.OptimizeResult()
                results.x = init_shift
                results.success = True
                results.message = 'Optimization not performed.'
                final_matrix = params_to_matrix(init_params)
            elif opt_method == "Auto":
                results = optimize.minimize(error_func, init_params)
                final_matrix = params_to_matrix(results.x)
            else:
                results = optimize.minimize(error_func, init_params, method = opt_method)
                final_matrix = params_to_matrix(results.x)

        return {'matrix': final_matrix, 'init': init_shift, 'opt_method': opt_method, \
                'reg_method': reg_method, 'results': results}

