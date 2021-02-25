#!/usr/bin/env python

import sys, numpy
from scipy.signal import convolve
from scipy.ndimage import zoom
from . import mmtiff

import cupy
from cupyx.scipy.ndimage import zoom as cupyx_zoom
from cupyx.scipy.signal import convolve as cupyx_convolve

class Lucy:
    def __init__ (self, psf_image, gpu_id = None, use_fft = False, save_memory = False):
        self.psf = psf_image.astype(numpy.float) / numpy.sum(psf_image)
        self.hat = numpy.flip(self.psf)
        self.gpu_id = gpu_id
        self.use_fft = use_fft
        self.save_memory = save_memory
        self.psf_fft = None
        self.hat_fft = None
        self.psf_fft_gpu = None
        self.hat_fft_gpu = None
        self.deconvolve = None # a pointer for function
        if self.gpu_id is None:
            if self.use_fft:
                self.deconvolve = self.deconvolve_cpu_fft
            else:
                self.deconvolve = self.deconvolve_cpu_mat
        else:
            device = cupy.cuda.Device(gpu_id)
            device.use()
            print("Turning on GPU: {0}, PCI-bus ID: {1}".format(self.gpu_id, device.pci_bus_id))
            if self.save_memory is False:
                print("Memory saving disabled. The GPU may give an out-of-memory error.")

            if self.use_fft:
                self.deconvolve = self.deconvolve_gpu_fft
            else:
                print("GPU version of the Matrix deconvolution is under construction.")
                self.deconvolve = self.deconvolve_gpu_mat

    def update_psf_fft (self, shape):
        if (self.psf_fft is not None) and (self.psf_fft.shape == shape):
            return

        # update psf and hat
        psf_resized = mmtiff.MMTiff.resize(self.psf, shape, center = True)
        if self.gpu_id is not None:
            # send psf images to the GPU
            if self.save_memory:
                psf_resized = cupy.array(psf_resized.astype(numpy.float32))
            else:
                psf_resized = cupy.array(psf_resized)

            # cupy seems to output complex64 for float32 input
            self.psf_fft_gpu = cupy.fft.fftn(cupy.fft.ifftshift(psf_resized))
            self.hat_fft_gpu = cupy.fft.fftn(cupy.fft.ifftshift(cupy.flip(psf_resized)))

            # copy to cpu
            self.psf_fft = cupy.asnumpy(self.psf_fft_gpu)
            self.hat_fft = cupy.asnumpy(self.hat_fft_gpu)
        else:
            if self.save_memory:
                print("Preparing complex64 psf to save memory.")
                self.psf_fft = numpy.fft.fftn(numpy.fft.ifftshift(psf_resized)).astype(numpy.complex64)
                self.hat_fft = numpy.fft.fftn(numpy.fft.ifftshift(numpy.flip(psf_resized))).astype(numpy.complex64)
            else:
                self.psf_fft = numpy.fft.fftn(numpy.fft.ifftshift(psf_resized))
                self.hat_fft = numpy.fft.fftn(numpy.fft.ifftshift(numpy.flip(psf_resized)))

    def deconvolve_cpu_mat (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if self.save_memory:
            orig_image = input_image.astype(numpy.float32)
        else:
            orig_image = input_image.astype(numpy.float)

        # zoom image, may be zoomed when z_zoom = 1.0 (float)
        if z_zoom != 1:
            orig_image = zoom(orig_image, (z_zoom, 1.0, 1.0))

        latent_est = orig_image.copy()
        for i in range(iterations):
            latent_est = latent_est * \
                convolve(orig_image / convolve(latent_est, self.psf, "same"), self.hat, "same")

        if zoom_result and z_zoom != 1:
            latent_est = zoom(latent_est, (1.0/z_zoom, 1.0, 1.0))

        return latent_est

    def deconvolve_cpu_fft (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if self.save_memory:
            orig_image = input_image.astype(numpy.float32)
        else:
            orig_image = input_image.astype(numpy.float)

        if z_zoom != 1:
            # zoom image, may be zoomed when z_zoom = 1.0 (float)
            orig_image = zoom(orig_image, (z_zoom, 1.0, 1.0))

        self.update_psf_fft(orig_image.shape)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / numpy.abs(numpy.fft.ifftn(self.psf_fft * numpy.fft.fftn(latent_est)))
            latent_est = latent_est * numpy.abs(numpy.fft.ifftn(self.hat_fft * numpy.fft.fftn(ratio)))

        if zoom_result and z_zoom != 1:
            latent_est = zoom(latent_est, (1.0/z_zoom, 1.0, 1.0))

        return latent_est

    def deconvolve_gpu_mat (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if self.save_memory:
            orig_image = cupy.array(input_image.astype(numpy.float32))
            psf_image = cupy.array(self.psf.astype(numpy.float32))
            hat_image = cupy.array(self.hat.astype(numpy.float32))
        else:
            orig_image = cupy.array(input_image.astype(numpy.float))
            psf_image = cupy.array(self.psf)
            hat_image = cupy.array(self.hat)

        # zoom image, may be zoomed when z_zoom = 1.0 (float)
        if z_zoom != 1:
            orig_image = cupyx_zoom(orig_image, (z_zoom, 1.0, 1.0), order = 1)

        latent_est = orig_image.copy()
        for i in range(iterations):
            latent_est = latent_est * \
                cupyx_convolve(orig_image / cupyx_convolve(latent_est, psf_image, "same"), \
                               hat_image, "same")

        if zoom_result and z_zoom != 1:
            latent_est = cupyx_zoom(latent_est, (1.0/z_zoom, 1.0, 1.0), order = 1)

        return cupy.asnumpy(latent_est)

    def deconvolve_gpu_fft (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if self.save_memory:
            orig_image = cupy.array(input_image.astype(numpy.float32))
        else:
            orig_image = cupy.array(input_image.astype(numpy.float))

        # zoom image, may be zoomed when z_zoom = 1.0 (float)
        if z_zoom != 1:
            orig_image = cupyx_zoom(orig_image, (z_zoom, 1.0, 1.0), order = 1)

        self.update_psf_fft(orig_image.shape)

        #orig_image = cupy.array(orig_image)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / cupy.abs(cupy.fft.ifftn(self.psf_fft_gpu * cupy.fft.fftn(latent_est)))
            latent_est = latent_est * cupy.abs(cupy.fft.ifftn(self.hat_fft_gpu * cupy.fft.fftn(ratio)))

        #latent_est = cupy.asnumpy(latent_est)
        if zoom_result and z_zoom != 1:
            latent_est = cupyx_zoom(latent_est, (1.0/z_zoom, 1.0, 1.0), order = 1)

        return cupy.asnumpy(latent_est)

