#!/usr/bin/env python

import sys
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from . import mmtiff
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

def turn_on_gpu (gpu_id):
    if gpu_id is None:
        return None
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)
    return device

class Lucy:
    def __init__ (self, psf_image, gpu_id = None):
        self.psf = psf_image.astype(np.float) / np.sum(psf_image)
        self.hat = np.flip(self.psf)
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

