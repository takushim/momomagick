#!/usr/bin/env python

import sys
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
registering_methods = ["Full", "Rigid-Zoom", "Rigid", "Drift", "XY", "POC", "None"]

def turn_on_gpu (gpu_id):
    return gpuimage.turn_on_gpu(gpu_id)

def find_method (name, method_list):
    try:
        lower_list = [method.lower() for method in method_list]
        name = method_list[lower_list.index(name.lower())]
    except ValueError:
        name = None

    return name

def window_mat (shape, window_func = None):
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

def xydrift_to_matrix_2d (params):
    return np.array([[1.0, 0.0, params[0]], [0.0, 1.0, params[1]], [0.0, 0.0, 1.0]])

def xydrift_to_matrix_3d (params):
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, params[0]], \
                     [0.0, 0.0, 1.0, params[1]], [0.0, 0.0, 0.0, 1.0]])

def drift_to_matrix_2d (params):
    return np.array([[1.0, 0.0, params[0]], [0.0, 1.0, params[1]], [0.0, 0.0, 1.0]])

def drift_to_matrix_3d (params):
    return np.array([[1.0, 0.0, 0.0, params[0]], [0.0, 1.0, 0.0, params[1]], \
                     [0.0, 0.0, 1.0, params[2]], [0.0, 0.0, 0.0, 1.0]])

def rbm_to_matrix_2d (params):
    return compose_matrix_2d(shift = params[0:2], rotation = params[2])

def rbm_to_matrix_3d (params):
    return compose_matrix_3d(shift = params[0:3], rotation = params[3:6])

def rbmzoom_to_matrix_2d (params):
    return compose_matrix_2d(shift = params[0:2], rotation = params[2], zoom = params[3:5])

def rbmzoom_to_matrix_3d (params):
    return compose_matrix_3d(shift = params[0:3], rotation = params[3:6], zoom = params[6:9])

def full_to_matrix_2d (params):
    return np.array([params[0:3], params[3:6], [0.0, 0.0, 1.0]])

def full_to_matrix_3d (params):
    return np.array([params[0:4], params[4:8], params[8:12], [0.0, 0.0, 0.0, 1.0]])

def compose_matrix_2d (shift = [0.0, 0.0], rotation = [0.0], zoom = [1.0, 1.0], shear = [0.0, 0.0]):
    rotation = np.array(rotation).item() * np.pi / 180
    rot_mat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation),  np.cos(rotation)]])
    return affines.compose(shift, rot_mat, zoom, shear)

def compose_matrix_3d (shift = [0.0, 0.0, 0.0], rotation = [0.0, 0.0, 0.0], zoom = [1.0, 1.0, 1.0], shear = [0.0, 0.0, 0.0]):
    rot_mat = euler.euler2mat(*(np.array(rotation) * np.pi / 180))
    return affines.compose(shift, rot_mat, zoom, shear)

def decompose_matrix (matrix):
    # interpret the affine matrix
    shift, rot_mat, zoom, shear = affines.decompose(matrix)
    if rot_mat.shape == (3, 3):
        rot_angles = np.array(euler.mat2euler(rot_mat)) / np.pi * 180
    elif rot_mat.shape == (2, 2):
        rot_angles = np.array([np.arccos(rot_mat[0, 0])]) / np.pi * 180
    else:
        print("Cannot analyze the roration matrix:", rot_mat)
        rot_angles = None

    return {'shift': shift, 'rotation_matrix': rot_mat, 'rotation_angles': rot_angles, \
            'zoom': zoom, 'shear': shear}

def offset_matrix (matrix, offset):
    if len(offset) == 2:
        matrix[0:2, 2] = matrix[0:2, 2] + np.array(offset)
    elif len(offset) == 3:
        matrix[0:3, 3] = matrix[0:3, 3] + np.array(offset)
    else:
        raise Exception("Cannot offset the matrix:", offset)
    return matrix

def params_dict_2d (shift = [0.0, 0.0], rotation = [0.0], zoom = [1.0, 1.0], shear = [0.0, 0.0]):
    params = {}
    params['shift'] = shift
    params['rotation'] = [np.array(rotation).item()]
    params['zoom'] = zoom
    params['shear'] = shear
    return params

def params_dict_3d (shift = [0.0, 0.0, 0.0], rotation = [0.0, 0.0, 0.0], zoom = [1.0, 1.0, 1.0], shear = [0.0, 0.0, 0.0]):
    params = {}
    params['shift'] = shift
    params['rotation'] = rotation
    params['zoom'] = zoom
    params['shear'] = shear
    return params

def register (ref_image, input_image, init_params = None, gpu_id = None, reg_method = "Full", opt_method = "Powell"):
    if init_params is None:
        # calculate POCs for pre-registration
        poc_register = Poc(ref_image, gpu_id = gpu_id)
        if reg_method == "POC":
            poc_result = poc_register.register_subpixel(input_image, opt_method = opt_method)
        else:
            poc_result = poc_register.register(input_image)
        poc_register = None

    # calculate an affine matrix for registration
    if len(input_image.shape) < 3:
        if init_params is None:
            init_params = params_dict_2d(shift = poc_result['shift'])
    elif input_image.shape[0] == 1:
        if init_params is None:
            init_params = params_dict_2d(shift = poc_result['shift'])
        ref_image = ref_image[0]
        input_image = input_image[0]
    else:
        if init_params is None:
            init_params = params_dict_3d(shift = poc_result['shift'])

    affine_register = Affine(ref_image, gpu_id = gpu_id)
    affine_result = affine_register.register(input_image, init_params, opt_method = opt_method, reg_method = reg_method)

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

        full_size = np.array(input_image.shape).astype(float)
        half_size = full_size / 2
        corr = max_val / (np.prod(half_size) / np.prod(full_size))

        return {'shift': shift, 'corr': corr}

    def register_subpixel (self, input_image, fit_size = 9, opt_method = "Powell"):
        opt_method = find_method(opt_method, optimizing_methods)

        poc_image = self.poc_image(input_image)
        max_pos = ndimage.maximum_position(poc_image)
        max_val = poc_image[max_pos]
        full_size = np.array(input_image.shape).astype(float)
        half_size = full_size / 2

        fit_shape = np.array([fit_size for s in input_image.shape])
        poc_fit = gpuimage.crop(poc_image, max_pos - fit_shape // 2, fit_shape)
        input_range = [np.arange(-(fit_size // 2), (fit_size + 1) // 2, 1) for s in input_image.shape]
        inputs = np.meshgrid(*input_range, indexing = 'ij')

        if self.gpu_id is None:
            def error_func (params):
                alpha = params[0]
                deltas = params[1:]
                matrix = alpha * np.prod(half_size) / np.prod(full_size) * np.ones(fit_shape)
                for i in range(len(input_image.shape)):
                    matrix = matrix * np.sinc((inputs[i] - deltas[i]) * half_size[i] / full_size[i]) \
                                    / np.sinc((inputs[i] - deltas[i]) / full_size[i])
                return np.sum((poc_fit - matrix) * (poc_fit - matrix))
        else:
            inputs = cp.array(inputs)
            poc_fit = cp.array(poc_fit)
            full_size = cp.array(full_size)
            half_size = cp.array(half_size)
            def error_func (params):
                alpha = params[0]
                deltas = params[1:]
                matrix = alpha * cp.prod(half_size) / cp.prod(full_size) * cp.ones(fit_shape)
                for i in range(len(input_image.shape)):
                    matrix = matrix * cp.sinc((inputs[i] - deltas[i]) * half_size[i] / full_size[i]) \
                                    / cp.sinc((inputs[i] - deltas[i]) / full_size[i])
                return cp.asnumpy(cp.sum((poc_fit - matrix) * (poc_fit - matrix)))

        init_params = [0.5] + [0.0] * len(input_image.shape)
        if opt_method == "None":
            results = optimize.OptimizeResult()
            results.x = init_params
            results.success = True
            results.message = 'Optimization not performed.'
        elif opt_method == "Auto":
            results = optimize.minimize(error_func, init_params)
        else:
            results = optimize.minimize(error_func, init_params, method = opt_method)

        init_corr = max_val / (np.prod(half_size) / np.prod(full_size))
        return {'shift': (max_pos - self.center + results.x[1:]), 'corr': results.x[0], \
                'init_shift': (max_pos - self.center), 'init_corr': init_corr, \
                'opt_method': opt_method, 'init_params': init_params, \
                'results': results}

class Affine:
    def __init__ (self, ref_image, window_func = np.hanning, gpu_id = None):
        self.window_mat = window_mat(ref_image.shape, window_func)
        self.ref_float = normalize(ref_image)
        self.gpu_id = gpu_id
        if gpu_id is not None:
            self.window_mat = cp.array(self.window_mat)
            self.ref_float = cp.array(self.ref_float)

    def register (self, input_image, init_params = None, opt_method = "Powell", reg_method = "Full"):
        reg_method = find_method(reg_method, registering_methods)
        opt_method = find_method(opt_method, optimizing_methods)

        if len(input_image.shape) == 2:
            if init_params is None:
                init_params = params_dict_2d()
            else:
                init_params = {key: list(value) for (key, value) in init_params.items()}

            init_matrix = compose_matrix_2d(shift = init_params['shift'], rotation = init_params['rotation'], \
                                            zoom = init_params['zoom'], shear = init_params['shear'])

            if reg_method == 'None' or reg_method == 'POC':
                opt_params = np.zeros(2)
                params_to_matrix = drift_to_matrix_2d
            elif reg_method == 'XY':
                opt_params = np.zeros(2)
                params_to_matrix = xydrift_to_matrix_2d
            elif reg_method == 'Drift':
                opt_params = np.zeros(2)
                params_to_matrix = drift_to_matrix_2d
            elif reg_method == 'Rigid':
                opt_params = np.zeros(3)
                params_to_matrix = rbm_to_matrix_2d
            elif reg_method == 'Rigid-Zoom':
                opt_params = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
                params_to_matrix = rbmzoom_to_matrix_2d
            elif reg_method == 'Full':
                opt_params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                params_to_matrix =full_to_matrix_2d
            else:
                raise Exception("Unknown registration method:", reg_method)
        elif len(input_image.shape) == 3:
            if init_params is None:
                init_params = params_dict_3d()
            else:
                init_params = {key: list(value) for (key, value) in init_params.items()}

            init_matrix = compose_matrix_3d(shift = init_params['shift'], rotation = init_params['rotation'], \
                                            zoom = init_params['zoom'], shear = init_params['shear'])

            if reg_method == 'None' or reg_method == 'POC':
                opt_params = np.zeros(3)
                params_to_matrix = drift_to_matrix_3d
            elif reg_method == 'XY':
                opt_params = np.zeros(3)
                params_to_matrix = xydrift_to_matrix_3d
            elif reg_method == 'Drift':
                opt_params = np.zeros(3)
                params_to_matrix = drift_to_matrix_3d
            elif reg_method == 'Rigid':
                opt_params = np.zeros(6)
                params_to_matrix = rbm_to_matrix_3d
            elif reg_method == 'Rigid-Zoom':
                opt_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                params_to_matrix = rbmzoom_to_matrix_3d
            elif reg_method == 'Full':
                opt_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                params_to_matrix = full_to_matrix_3d
            else:
                raise Exception("Unknown registration method:", reg_method)
        else:
            raise Exception("Input images must be 2D or 3D grayscale.")

        if self.gpu_id is None:
            image_float = normalize(input_image)
            def error_func (params):
                matrix = params_to_matrix(params)
                trans_float = ndimage.affine_transform(image_float, np.dot(matrix, init_matrix)) * self.window_mat
                error = np.sum((self.ref_float - trans_float) * (self.ref_float - trans_float))
                return error
        else:
            image_float = cp.array(normalize(input_image))
            init_matrix_gpu = cp.array(init_matrix)
            def error_func (params):
                matrix = cp.array(params_to_matrix(params))
                trans_float = cpimage.affine_transform(image_float, cp.dot(matrix, init_matrix_gpu)) * self.window_mat
                error = cp.asnumpy(cp.sum((self.ref_float - trans_float) * (self.ref_float - trans_float)))
                return error

        if reg_method == "None" or reg_method == 'POC':
            results = optimize.OptimizeResult()
            results.x = opt_params
            results.success = True
            results.message = 'Registration not performed.'
            final_matrix = np.dot(params_to_matrix(opt_params), init_matrix)
        else:
            if opt_method == "None":
                results = optimize.OptimizeResult()
                results.x = opt_params
                results.success = True
                results.message = 'Optimization not performed.'
                final_matrix = np.dot(params_to_matrix(opt_params), init_matrix)
            elif opt_method == "Auto":
                results = optimize.minimize(error_func, opt_params)
                final_matrix = np.dot(params_to_matrix(results.x), init_matrix)
            else:
                results = optimize.minimize(error_func, opt_params, method = opt_method)
                final_matrix = np.dot(params_to_matrix(results.x), init_matrix)

        return {'matrix': final_matrix, 'init_params': init_params, 'init_matrix': init_matrix, \
                'opt_params': results.x, 'opt_matrix': params_to_matrix(results.x),
                'opt_method': opt_method, 'reg_method': reg_method, 'results': results}

