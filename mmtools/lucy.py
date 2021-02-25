#!/usr/bin/env python

import sys, numpy
from scipy.signal import convolve
from scipy.ndimage import zoom
from . import mmtiff

# global variables to store modules and functions
g_cupy = None
g_cupyx_zoom = None
g_cupyx_convolve = None

class Lucy:
    def __init__ (self, psf_image, gpu_id = None, use_matrix = False, save_memory = False):
        self.psf = psf_image.astype(numpy.float) / numpy.sum(psf_image)
        self.hat = numpy.flip(self.psf)
        self.gpu_id = gpu_id
        self.use_matrix = use_matrix
        self.save_memory = save_memory
        self.psf_fft = None
        self.hat_fft = None
        self.psf_fft_gpu = None
        self.hat_fft_gpu = None
        self.deconvolve = None # a pointer for function
        if self.gpu_id is None:
            if self.use_matrix:
                self.deconvolve = self.deconvolve_cpu_mat
            else:
                self.deconvolve = self.deconvolve_cpu_fft
        else:
            import cupy
            from cupyx.scipy.ndimage import zoom as cupyx_zoom
            from cupyx.scipy.signal import convolve as cupyx_convolve
            # export modules to the global scope
            global g_cupy
            global g_cupyx_zoom
            global g_cupyx_convolve
            g_cupy = cupy
            g_cupyx_zoom = cupyx_zoom
            g_cupyx_convolve = cupyx_convolve

            device = g_cupy.cuda.Device(gpu_id)
            device.use()
            print("Turning on GPU: {0}, PCI-bus ID: {1}".format(self.gpu_id, device.pci_bus_id))
            if self.save_memory is False:
                print("Memory saving disabled. The GPU may give an out-of-memory error.")

            if self.use_matrix:
                print("GPU version of the Matrix deconvolution is under construction.")
                self.deconvolve = self.deconvolve_gpu_mat
            else:
                self.deconvolve = self.deconvolve_gpu_fft

    def update_psf_fft (self, shape):
        if (self.psf_fft is not None) and (self.psf_fft.shape == shape):
            return

        # update psf and hat
        psf_resized = mmtiff.MMTiff.resize(self.psf, shape, center = True)
        if self.gpu_id is not None:
            # send psf images to the GPU
            if self.save_memory:
                psf_resized = g_cupy.array(psf_resized.astype(numpy.float32))
            else:
                psf_resized = g_cupy.array(psf_resized)

            # cupy seems to output complex64 for float32 input
            self.psf_fft_gpu = g_cupy.fft.fftn(g_cupy.fft.ifftshift(psf_resized))
            self.hat_fft_gpu = g_cupy.fft.fftn(g_cupy.fft.ifftshift(g_cupy.flip(psf_resized)))

            # copy to cpu
            self.psf_fft = g_cupy.asnumpy(self.psf_fft_gpu)
            self.hat_fft = g_cupy.asnumpy(self.hat_fft_gpu)
        else:
            if self.save_memory:
                print("Preparing complex64 psf to save memory.")
                self.psf_fft = numpy.fft.fftn(numpy.fft.ifftshift(psf_resized)).astype(numpy.complex64)
                self.hat_fft = numpy.fft.fftn(numpy.fft.ifftshift(numpy.flip(psf_resized))).astype(numpy.complex64)
            else:
                self.psf_fft = numpy.fft.fftn(numpy.fft.ifftshift(psf_resized))
                self.hat_fft = numpy.fft.fftn(numpy.fft.ifftshift(numpy.flip(psf_resized)))

    def deconvolve_cpu_mat (self, input_image, iterations = 10, z_scale = 1, keep_z_scale = False):
        if self.save_memory:
            orig_image = input_image.astype(numpy.float32)
        else:
            orig_image = input_image.astype(numpy.float)

        # scale image, may be scaled when z_scale = 1.0 (float)
        if z_scale != 1:
            orig_image = zoom(orig_image, (z_scale, 1.0, 1.0))

        latent_est = orig_image.copy()
        for i in range(iterations):
            latent_est = latent_est * \
                convolve(orig_image / convolve(latent_est, self.psf, "same"), self.hat, "same")

        if keep_z_scale is False and z_scale != 1:
            latent_est = zoom(latent_est, (1.0/z_scale, 1.0, 1.0))

        return latent_est

    def deconvolve_cpu_fft (self, input_image, iterations = 10, z_scale = 1, keep_z_scale = False):
        if self.save_memory:
            orig_image = input_image.astype(numpy.float32)
        else:
            orig_image = input_image.astype(numpy.float)

        if z_scale != 1:
            # scale image, may be scaled when z_scale = 1.0 (float)
            orig_image = zoom(orig_image, (z_scale, 1.0, 1.0))

        self.update_psf_fft(orig_image.shape)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / numpy.abs(numpy.fft.ifftn(self.psf_fft * numpy.fft.fftn(latent_est)))
            latent_est = latent_est * numpy.abs(numpy.fft.ifftn(self.hat_fft * numpy.fft.fftn(ratio)))

        if keep_z_scale is False and z_scale != 1:
            latent_est = zoom(latent_est, (1.0/z_scale, 1.0, 1.0))

        return latent_est

    def deconvolve_gpu_mat (self, input_image, iterations = 10, z_scale = 1, keep_z_scale = False):
        if self.save_memory:
            orig_image = g_cupy.array(input_image.astype(numpy.float32))
            psf_image = g_cupy.array(self.psf.astype(numpy.float32))
            hat_image = g_cupy.array(self.hat.astype(numpy.float32))
        else:
            orig_image = g_cupy.array(input_image.astype(numpy.float))
            psf_image = g_cupy.array(self.psf)
            hat_image = g_cupy.array(self.hat)

        # scale image, may be scaled when z_scale = 1.0 (float)
        if z_scale != 1:
            orig_image = g_cupyx_zoom(orig_image, (z_scale, 1.0, 1.0), order = 1)

        latent_est = orig_image.copy()
        for i in range(iterations):
            latent_est = latent_est * \
                g_cupyx_convolve(orig_image / g_cupyx_convolve(latent_est, psf_image, "same"), \
                                 hat_image, "same")

        if keep_z_scale is False and z_scale != 1:
            latent_est = g_cupyx_zoom(latent_est, (1.0/z_scale, 1.0, 1.0), order = 1)

        return g_cupy.asnumpy(latent_est)

    def deconvolve_gpu_fft (self, input_image, iterations = 10, z_scale = 1, keep_z_scale = False):
        if self.save_memory:
            orig_image = g_cupy.array(input_image.astype(numpy.float32))
        else:
            orig_image = g_cupy.array(input_image.astype(numpy.float))

        # scale image, may be scaled when z_scale = 1.0 (float)
        if z_scale != 1:
            orig_image = g_cupyx_zoom(orig_image, (z_scale, 1.0, 1.0), order = 1)

        self.update_psf_fft(orig_image.shape)

        #orig_image = cupy.array(orig_image)
        latent_est = orig_image.copy()
        for i in range(iterations):
            ratio = orig_image / g_cupy.abs(g_cupy.fft.ifftn(self.psf_fft_gpu * g_cupy.fft.fftn(latent_est)))
            latent_est = latent_est * g_cupy.abs(g_cupy.fft.ifftn(self.hat_fft_gpu * g_cupy.fft.fftn(ratio)))

        #latent_est = cupy.asnumpy(latent_est)
        if keep_z_scale is False and z_scale != 1:
            latent_est = g_cupyx_zoom(latent_est, (1.0/z_scale, 1.0, 1.0), order = 1)

        return g_cupy.asnumpy(latent_est)

