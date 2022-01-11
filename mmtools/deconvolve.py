#!/usr/bin/env python

import numpy as np
from . import stack
try:
    import cupy as cp
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    return stack.turn_on_gpu(gpu_id)

def deconvolve (input_image, psf_image, iterations = 10, gpu_id = None):
    if gpu_id is None:
        return deconvolve_cpu(input_image, psf_image, iterations = iterations)
    else:
        return deconvolve_gpu(input_image, psf_image, iterations = iterations)

def deconvolve_cpu (input_image, psf_image, iterations = 10):
    orig_image = input_image.astype(float)

    if isinstance(psf_image, list):
        psf_images = psf_image
    else:
        psf_images = [psf_image]
    
    psf_images = [image.astype(float) / np.sum(image) for image in psf_images]
    psf_images = [stack.resize(image, orig_image.shape, centering = True) for image in psf_images]

    psf_ffts = [np.fft.fftn(np.fft.ifftshift(image)) for image in psf_images]
    hat_ffts = [np.fft.fftn(np.fft.ifftshift(np.flip(image))) for image in psf_images]

    latent_est = orig_image.copy()
    for iteration in range(iterations):
        for index in range(len(psf_images)):
            ratio = orig_image / np.abs(np.fft.ifftn(psf_ffts[index] * np.fft.fftn(latent_est)))
            latent_est = latent_est * np.abs(np.fft.ifftn(hat_ffts[index] * np.fft.fftn(ratio)))

    return latent_est

def deconvolve_gpu (input_image, psf_image, iterations = 10):
    orig_image = cp.array(input_image.astype(float))

    if isinstance(psf_image, list):
        psf_images = psf_image
    else:
        psf_images = [psf_image]
    
    psf_images = [image.astype(float) / np.sum(image) for image in psf_images]
    psf_images = [stack.resize(image, orig_image.shape, centering = True) for image in psf_images]
    psf_images = [cp.array(image) for image in psf_images]

    psf_ffts = [cp.fft.fftn(cp.fft.ifftshift(image)) for image in psf_images]
    hat_ffts = [cp.fft.fftn(cp.fft.ifftshift(cp.flip(image))) for image in psf_images]

    latent_est = orig_image.copy()
    for iteration in range(iterations):
        for index in range(len(psf_images)):
            ratio = orig_image / cp.abs(cp.fft.ifftn(psf_ffts[index] * cp.fft.fftn(latent_est)))
            latent_est = latent_est * cp.abs(cp.fft.ifftn(hat_ffts[index] * cp.fft.fftn(ratio)))

    return cp.asnumpy(latent_est)
