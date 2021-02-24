#!/usr/bin/env python

import sys, numpy
from scipy.signal import convolve
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import shift
import cupy
from cupyx.scipy.ndimage import zoom as cupyx_zoom
#from cupyx.scipy.ndimage import convolve as cupyx_convolve

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
        if self.gpu_id is not None:
            device = cupy.cuda.Device(gpu_id)
            device.use()
            print("Turning on GPU: {0}, PCI-bus ID: {1}".format(self.gpu_id, device.pci_bus_id))
            if self.use_fft is False:
                print("GPU version of the Matrix deconvolution is distabled. FFT used.")
            if self.save_memory is False:
                print("Memory saving disabled. The GPU may give an out-of-memory error.")

    def update_psf_fft (self, shape):
        if (self.psf_fft is None) or (self.psf_fft.shape != shape):
            psf_resized = numpy.zeros(shape, dtype = numpy.float)
            limits = numpy.minimum(shape, self.psf.shape)
            shifts = (numpy.array(shape) - numpy.array(self.psf.shape)) // 2
            psf_resized[0:limits[0], 0:limits[1], 0:limits[2]] = self.psf[0:limits[0], 0:limits[1], 0:limits[2]]
            psf_resized = shift(psf_resized, shifts)
            # send psf images to the GPU
            if self.gpu_id is not None:
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

    def deconvolve (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if input_image.ndim < 3:
            print("Zoom disabled since image dimension == {0}".format(input_image.ndim))
            z_zoom = 1
            zoom_result = False

        if self.gpu_id is None:
            if self.use_fft:
                latent_est = self.deconvolve_cpu_fft(input_image, iterations, z_zoom, zoom_result)
            else:
                latent_est = self.deconvolve_cpu_mat(input_image, iterations, z_zoom, zoom_result)
        else:
            if self.use_fft:
                latent_est = self.deconvolve_gpu_fft(input_image, iterations, z_zoom, zoom_result)
            else:
                latent_est = self.deconvolve_gpu_fft(input_image, iterations, z_zoom, zoom_result)

        return latent_est

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
        # broken!
        raise Exception("Does not work on gpu!")

        orig_image = input_image.astype(numpy.float32)
        if z_zoom != 1:
            # zoom image, may be zoomed when z_zoom = 1.0 (float)
            orig_image_gpu = cupyx_zoom(orig_image_gpu, (z_zoom, 1.0, 1.0), order = 1)

        latent_est_gpu = orig_image_gpu.copy()
        temp_image_gpu = cupy.zeros_like(orig_image_gpu, dtype = numpy.float32)

        for i in range(iterations):
            print("Iteration:", i)
            print(latent_est_gpu.shape)
            print(temp_image_gpu.shape)
            temp_image_gpu = cupyx_convolve(latent_est_gpu, self.psf_image_gpu, mode = "constant")
            print("Convolution 1")
            temp_image_gpu = orig_image_gpu / temp_image_gpu
            print("Division")
            cupyx_convolve(latent_est_gpu, self.psf_hat_gpu, output = temp_image_gpu, mode = "constant")
            print("Convolution 2")
            latent_est_gpu = latent_est_gpu * temp_image_gpu
            print("Done")
            #latent_est_gpu = latent_est_gpu * \
            #    cupyx_convolve(orig_image_gpu / cupyx_convolve(latent_est_gpu, self.psf_image_gpu, \
            #                                                  mode = "constant"), \
            #             self.psf_hat_gpu, mode = "constant")

        if zoom_result and z_zoom != 1:
            latent_est_gpu = cupyx_zoom(latent_est_gpu, (1.0/z_zoom, 1.0, 1.0), order = 1)

        return cupy.asnumpy(latent_est_gpu)

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

