#!/usr/bin/env python

import sys
import numpy as np
from . import gpuimage
try:
    import cupy as cp
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    return gpuimage.turn_on_gpu(gpu_id)

class Lucy:
    def __init__ (self, psf_image, gpu_id = None):
        self.gpu_id = gpu_id
        if gpu_id is None:
            self.deconvolve = self.deconvolve_cpu
        else:
            self.deconvolve = self.deconvolve_gpu

        if isinstance(psf_image, list):
            self.psf_images = psf_image
        else:
            self.psf_images = [psf_image]

        self.psf_images = [image.astype(float) / np.sum(image) for image in self.psf_images]
        self.psf_ffts = [None for i in self.psf_images]
        self.hat_ffts = [None for i in self.psf_images]

    def update_psf_fft (self, shape):
        for index in range(len(self.psf_images)):
            if (self.psf_ffts[index] is not None) and (self.psf_ffts[index].shape == shape):
                continue
            # update psf and hat
            psf_resized = gpuimage.resize(self.psf_images[index], shape, center = True)
            if self.gpu_id is None:
                self.psf_ffts[index] = np.fft.fftn(np.fft.ifftshift(psf_resized))
                self.hat_ffts[index] = np.fft.fftn(np.fft.ifftshift(np.flip(psf_resized)))
            else:
                psf_resized = cp.array(psf_resized)
                self.psf_ffts[index] = cp.fft.fftn(cp.fft.ifftshift(psf_resized))
                self.hat_ffts[index] = cp.fft.fftn(cp.fft.ifftshift(cp.flip(psf_resized)))

    def deconvolve_cpu (self, input_image, iterations = 10):
        orig_image = input_image.astype(float)
        self.update_psf_fft(orig_image.shape)
        latent_est = orig_image.copy()
        for iteration in range(iterations):
            for index in range(len(self.psf_images)):
                ratio = orig_image / np.abs(np.fft.ifftn(self.psf_ffts[index] * np.fft.fftn(latent_est)))
                latent_est = latent_est * np.abs(np.fft.ifftn(self.hat_ffts[index] * np.fft.fftn(ratio)))

        return latent_est

    def deconvolve_gpu (self, input_image, iterations = 10):
        orig_image = cp.array(input_image.astype(float))
        self.update_psf_fft(orig_image.shape)

        latent_est = orig_image.copy()
        for iteration in range(iterations):
            for index in range(len(self.psf_images)):
                ratio = orig_image / cp.abs(cp.fft.ifftn(self.psf_ffts[index] * cp.fft.fftn(latent_est)))
                latent_est = latent_est * cp.abs(cp.fft.ifftn(self.hat_ffts[index] * cp.fft.fftn(ratio)))

        return cp.asnumpy(latent_est)
