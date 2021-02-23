#!/usr/bin/env python

import sys, numpy
from scipy.signal import convolve
from scipy.ndimage import zoom

class Lucy:
    def __init__ (self, psf_image, gpu_id = None):
        self.psf_image = psf_image.astype(numpy.float) / numpy.sum(psf_image)
        self.psf_hat = numpy.flip(self.psf_image)
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            print("Turning on GPU {0}".format(self.gpu_id))
            import cupy
            self.gpu_device = cupy.cuda.Device(gpu_id)
            self.gpu_device.use()
            print("Device PCI bus id:", self.gpu_device.pci_bus_id)
            print("Sending psf images to GPU {0}".format(self.gpu_id))
            self.psf_image_gpu = cupy.asarray(self.psf_image)
            self.psf_hat_gpu = cupy.asarray(self.psf_hat)
            print("Free GPU memory bytes:", self.gpu_device.mem_info)

    def deconvolve (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        if input_image.ndim < 3:
            print("Zoom disabled since image dimension == {0}".format(input_image.ndim))
            z_zoom = 1
            zoom_result = False

        if self.gpu_id is None:
            latent_est = self.deconvolve_cpu(input_image, iterations, z_zoom, zoom_result)
        else:
            latent_est = self.deconvolve_gpu(input_image, iterations, z_zoom, zoom_result)

        return latent_est

    def deconvolve_cpu (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        orig_image = input_image.astype(numpy.float)
        if z_zoom != 1:
            # zoom image, may be zoomed when z_zoom = 1.0 (float)
            orig_image = zoom(orig_image, (z_zoom, 1.0, 1.0))

        latent_est = orig_image.copy()
        for i in range(iterations):
            latent_est = latent_est * \
                convolve(orig_image / convolve(latent_est, self.psf_image, "same"), \
                         self.psf_hat, "same")

        if zoom_result and z_zoom != 1:
            latent_est = zoom(latent_est, (1.0/z_zoom, 1.0, 1.0))

        return latent_est

    def deconvolve_gpu (self, input_image, iterations = 10, z_zoom = 1, zoom_result = False):
        import cupy
        from cupyx.scipy.ndimage import zoom as cupyx_zoom
        from cupyx.scipy.ndimage import convolve as cupyx_convolve
        #import sigpy

        orig_image_gpu = cupy.asarray(input_image.astype(numpy.float))
        if z_zoom != 1:
            # zoom image, may be zoomed when z_zoom = 1.0 (float)
            orig_image_gpu = cupyx_zoom(orig_image_gpu, (z_zoom, 1.0, 1.0), order = 1)

        latent_est_gpu = orig_image_gpu.copy()
        temp_image_gpu = cupy.zeros_like(orig_image_gpu)

        for i in range(iterations):
            print("Iteration:", i)
            #temp_image_gpu = sigpy.convolve(latent_est_gpu, self.psf_image_gpu, mode = 'same')
            cupyx_convolve(latent_est_gpu, self.psf_image_gpu, output = temp_image_gpu, mode = "constant")
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

