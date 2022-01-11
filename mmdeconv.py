#!/usr/bin/env python

import argparse, tifffile
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
iterations = 10
gpu_id = None
log_level = 'INFO'

parser = argparse.ArgumentParser(description = 'Deconvolve images using the Richardson-Lucy algorhythm', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='Output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='Psf image (current -> system folder). None: {0} or {1}'.format(psf_iso, psf_noniso))

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
logger.info("Loading image: {0}".format(input_filename))
input_stack = stack.Stack(input_filename)

# load psf image
if psf_filename is None:
    if scale_isometric:
        psf_path = Path(psf_folder).joinpath(psf_iso)
    else:
        psf_path = Path(psf_folder).joinpath(psf_noniso)
else:
    if Path(psf_filename).exists():
        psf_path = Path(psf_filename)
    else:
        psf_path = Path(psf_folder).joinpath(psf_filename)

logger.info("PSF image: {0}".format(psf_path))
psf_image = tifffile.imread(psf_path)

# setting image scale
pixel_orig = input_stack.pixel_um
if scale_isometric:
    logger.info("Scaling image: {0}".format(min(pixel_orig)))
    input_stack.scale_by_pixelsize(min(pixel_orig), gpu_id = gpu_id)

# deconvolution
def deconvolve_image (image):
    return deconvolve.deconvolve(image, psf_image, iterations = iterations, gpu_id = gpu_id)

logger.info("Deconvolution started")
input_stack.apply_all(deconvolve_image, progress = True)

if restore_scale:
    logger.info("Restoring scale: {0}".format(pixel_orig))
    input_stack.scale_by_pixelsize(pixel_orig, gpu_id = gpu_id)

# output in the ImageJ format, dimensions should be in TZCYX order
logger.info("Saving image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename)
