#!/usr/bin/env python

import argparse
import numpy as np
from pathlib import Path
from mmtools import stack, deconvolve, log

# defaults
psf_folder = Path(__file__).parent.joinpath('psf')
input_filename = None
output_filename = None
output_suffix = '_deconv.tif'
psf_filename = None
psf_iso = 'dispim_iso_bw.tif'
psf_noniso = 'dispim_0.5um_bw.tif'
psf_2d = 'dispim_2d_bw.tif'
iterations = 10
gpu_id = None
log_level = 'INFO'

parser = argparse.ArgumentParser(description = 'Deconvolve images using the Richardson-Lucy algorhythm', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='Output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='Name of psf image in the current or system folders. None: {0}'.format([psf_iso, psf_noniso, psf_2d]))

parser.add_argument('-i', '--iterations', type = int, default = iterations, \
                    help='Number of iterations')

parser.add_argument('-s', '--scale-isometric', action = 'store_true', \
                    help='Scale the image to make it isometric.')

parser.add_argument('-r', '--restore-scale', action = 'store_true', \
                    help='Restore the z scale of image after deconvolution')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-L', '--log-level', default = log_level, \
                    help='Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL')

parser.add_argument('input_file', default = input_filename, \
                    help='Image file to deconvolve')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# defaults
iterations = args.iterations
psf_filename = args.psf_image
scale_isometric = args.scale_isometric
restore_scale = args.restore_scale
gpu_id = args.gpu_id
input_filename = args.input_file
if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
stack.turn_on_gpu(gpu_id)

# load input image
logger.info("Loading image: {0}.".format(input_filename))
input_stack = stack.Stack(input_filename)

# load psf image
if psf_filename is None:
    if input_stack.z_count > 1:
        if scale_isometric:
            psf_path = Path(psf_folder).joinpath(psf_iso)
        else:
            psf_path = Path(psf_folder).joinpath(psf_noniso)
    else:
        psf_path = Path(psf_folder).joinpath(psf_2d)
else:
    if Path(psf_filename).exists():
        psf_path = Path(psf_filename)
    else:
        psf_path = Path(psf_folder).joinpath(psf_filename)

logger.info("PSF image: {0}.".format(psf_path))
psf_stack = stack.Stack(psf_path)
psf_image = psf_stack.image_array[0, 0]

# setting image scale
pixel_orig = input_stack.voxel_um
if scale_isometric:
    logger.info("Scaling image to be isometric. Pixel-size: {0}.".format(min(pixel_orig)))
    input_stack.scale_isometric(gpu_id = gpu_id)

if np.allclose(input_stack.voxel_um, psf_stack.voxel_um, atol = 1e-2) == False:
    logger.warning("Pixel sizes are different. Image: {0}. PSF: {1}.".format(input_stack.voxel_um, psf_stack.voxel_um))

# deconvolution
if input_stack.z_count > 1:
    def deconvolve_image (image, t_index, c_index):
        return deconvolve.deconvolve(image, psf_image, iterations = iterations, gpu_id = gpu_id)
else:
    def deconvolve_image (image, t_index, c_index):
        return deconvolve.deconvolve(image[0], psf_image[0], iterations = iterations, gpu_id = gpu_id)[np.newaxis]

logger.info("Deconvolution started.")
input_stack.apply_all(deconvolve_image, progress = True)

if restore_scale:
    logger.info("Restoring scale: {0}.".format(pixel_orig))
    input_stack.scale_by_pixelsize(pixel_orig, gpu_id = gpu_id)

# output in the ImageJ format, dimensions should be in TZCYX order
logger.info("Saving image: {0}.".format(output_filename))
input_stack.save_ome_tiff(output_filename, dtype = np.float32)
