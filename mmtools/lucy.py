#!/usr/bin/env python

import sys
import numpy as np
from . import mmtiff, register
try:
    import cupy as cp
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    return register.turn_on_gpu(gpu_id)

class Lucy:
    def __init__ (self, psf_image, gpu_id = None):
        self.psf = psf_image.astype(np.float) / np.sum(psf_image)
        self.gpu_id = gpu_id
        self.psf_fft = None
        self.hat_fft = None
        if gpu_id is None:
            self.deconvolve = self.deconvolve_cpu
        else:
            self.deconvolve = self.deconvolve_gpu

    def update_psf_fft (self, shape):
        if (self.psf_fft is not None) and (self.psf_fft.shape == shape):
            return
        # update psf and hat
        psf_resized = mmtiff.resize(self.psf, shape, center = True)
        if self.gpu_id is  None:
            self.psf_fft = np.fft.fftn(np.fft.ifftshift(psf_resized))
            self.hat_fft = np.fft.fftn(np.fft.ifftshift(np.flip(psf_resized)))
        else:
            psf_resized = cp.array(psf_resized)
            self.psf_fft = cp.fft.fftn(cp.fft.ifftshift(psf_resized))
            self.hat_fft = cp.fft.fftn(cp.fft.ifftshift(cp.flip(psf_resized)))

    def deconvolve_cpu (self, input_image, iterations = 10):
        orig_image = input_image.astype(float)
        self.update_psf_fft(orig_image.shape)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / np.abs(np.fft.ifftn(self.psf_fft * np.fft.fftn(latent_est)))
            latent_est = latent_est * np.abs(np.fft.ifftn(self.hat_fft * np.fft.fftn(ratio)))

        return latent_est

    def deconvolve_gpu (self, input_image, iterations = 10):
        orig_image = cp.array(input_image.astype(float))
        self.update_psf_fft(orig_image.shape)

        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / cp.abs(cp.fft.ifftn(self.psf_fft * cp.fft.fftn(latent_est)))
            latent_est = latent_est * cp.abs(cp.fft.ifftn(self.hat_fft * cp.fft.fftn(ratio)))

        return cp.asnumpy(latent_est)

class LucyDual:
    def __init__ (self, psf_image, angle = 0, gpu_id = None):
        self.psf = psf_image.astype(np.float) / np.sum(psf_image)
        self.psf_sub = register.z_rotate(psf_image, angle = angle, gpu_id = gpu_id)
        self.gpu_id = gpu_id
        self.psf_fft = None
        self.hat_fft = None
        self.psf_fft_sub = None
        self.hat_fft_sub = None
        if gpu_id is None:
            self.deconvolve = self.deconvolve_cpu
        else:
            self.deconvolve = self.deconvolve_gpu

    def update_psf_fft (self, shape):
        if (self.psf_fft is not None) and (self.psf_fft.shape == shape) and \
           (self.psf_fft_sub is not None) and (self.psf_fft_sub.shape == shape):
            return

        # update psf and hat
        psf_resized = mmtiff.resize(self.psf, shape, center = True)
        psf_resized_sub = mmtiff.resize(self.psf_sub, shape, center = True)
        if self.gpu_id is  None:
            self.psf_fft = np.fft.fftn(np.fft.ifftshift(psf_resized))
            self.hat_fft = np.fft.fftn(np.fft.ifftshift(np.flip(psf_resized)))
            self.psf_fft_sub = np.fft.fftn(np.fft.ifftshift(psf_resized_sub))
            self.hat_fft_sub = np.fft.fftn(np.fft.ifftshift(np.flip(psf_resized_sub)))
        else:
            psf_resized = cp.array(psf_resized)
            psf_resized_sub = cp.array(psf_resized_sub)
            self.psf_fft = cp.fft.fftn(cp.fft.ifftshift(psf_resized))
            self.hat_fft = cp.fft.fftn(cp.fft.ifftshift(cp.flip(psf_resized)))
            self.psf_fft_sub = cp.fft.fftn(cp.fft.ifftshift(psf_resized_sub))
            self.hat_fft_sub = cp.fft.fftn(cp.fft.ifftshift(cp.flip(psf_resized_sub)))

    def deconvolve_cpu (self, input_image, iterations = 10):
        orig_image = input_image.astype(float)
        self.update_psf_fft(orig_image.shape)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / np.abs(np.fft.ifftn(self.psf_fft * np.fft.fftn(latent_est)))
            latent_est = latent_est * np.abs(np.fft.ifftn(self.hat_fft * np.fft.fftn(ratio)))

            ratio = orig_image / np.abs(np.fft.ifftn(self.psf_fft_sub * np.fft.fftn(latent_est)))
            latent_est = latent_est * np.abs(np.fft.ifftn(self.hat_fft_sub * np.fft.fftn(ratio)))

        return latent_est

    def deconvolve_gpu (self, input_image, iterations = 10):
        orig_image = cp.array(input_image.astype(float))
        self.update_psf_fft(orig_image.shape)

        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / cp.abs(cp.fft.ifftn(self.psf_fft * cp.fft.fftn(latent_est)))
            latent_est = latent_est * cp.abs(cp.fft.ifftn(self.hat_fft * cp.fft.fftn(ratio)))

            ratio = orig_image / cp.abs(cp.fft.ifftn(self.psf_fft_sub * cp.fft.fftn(latent_est)))
            latent_est = latent_est * cp.abs(cp.fft.ifftn(self.hat_fft_sub * cp.fft.fftn(ratio)))

        return cp.asnumpy(latent_est)

