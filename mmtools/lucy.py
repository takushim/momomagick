#!/usr/bin/env python

import sys, numpy
from scipy.signal import convolve
from scipy.ndimage import zoom

class Lucy:
    def __init__ (self, psf_image, gpu_id = None):
        self.psf_image = psf_image.astype(numpy.float) / numpy.sum(psf_image)
        if gpu_id is None:
            self.deconvolve = self.deconvolve_cpu
        else:
            print("Turning on GPU {0}".format(gpu_id))
            import cupy
            cupy.cuda.Device(gpu_id).use()
            print("Free GPU memory bytes:", cupy.get_default_memory_pool().free_bytes)
            self.deconvolve = self.deconvolve_gpu

    def deconvolve_cpu (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if input_image.ndim < 3:
            print("Zoom disabled since image dimension == {0}".format(input_image.ndim))
            z_zoom = 1
            zoom_result = False

        # zoom image, may be zoomed when z_zoom = 1.0 (float)
        if z_zoom == 1:
            orig_image = input_image.astype(numpy.float).copy()
        else:
            orig_image = zoom(input_image.astype(numpy.float), (z_zoom, 1.0, 1.0))

        latent_est = orig_image.copy()
        psf_hat = numpy.flip(self.psf_image)
        for i in range(iterations):
            latent_est = latent_est * \
                convolve(orig_image / convolve(latent_est, self.psf_image, "same"), psf_hat, "same")

        if zoom_result and z_zoom != 1:
            latent_est = zoom(latent_est, (1.0/z_zoom, 1.0, 1.0))

        return latent_est

    def deconvolve_gpu (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if input_image.ndim < 3:
            print("Zoom disabled since image dimension == {0}".format(input_image.ndim))
            z_zoom = 1
            zoom_result = False

        # zoom image, may be zoomed when z_zoom = 1.0 (float)
        if z_zoom == 1:
            orig_image = input_image.astype(numpy.float).copy()
        else:
            orig_image = zoom(input_image.astype(numpy.float), (z_zoom, 1.0, 1.0))

        latent_est = orig_image.copy()
        psf_hat = numpy.flip(self.psf_image)
        for i in range(iterations):
            latent_est = latent_est * \
                convolve(orig_image / convolve(latent_est, self.psf_image, "same"), psf_hat, "same")

        if zoom_result and z_zoom != 1:
            latent_est = zoom(latent_est, (1.0/z_zoom, 1.0, 1.0))

        return latent_est

