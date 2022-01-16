#!/usr/bin/env python

import argparse
import numpy as np
from pathlib import Path
from progressbar import progressbar
from mmtools import stack, register, deconvolve, gpuimage, log

# default values
gpu_id = None
input_filename = None
output_filename = None
output_suffix = '_fusion.tif'
main_channel = 0
sub_channel = None
sub_rotation = 0
keep_channels = False
iterations = 0
reg_method = 'Full'
reg_method_list = register.registering_methods
opt_method = "Powell"
opt_method_list = register.optimizing_methods
psf_folder = Path(__file__).parent.joinpath('psf')
psf_filename = 'dispim_iso.tif'
log_level = 'INFO'


# parse arguments
parser = argparse.ArgumentParser(description='Fusion diSPIM images from two paths and deconvolve them.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='Output TIFF file ([basename0]{0} by default)'.format(output_suffix))

parser.add_argument('-m', '--main-channel', type = int, default = main_channel, \
                    help='Channel to map the XY plane of output')

parser.add_argument('-s', '--sub-channel', type = int, default = sub_channel, \
                    help='Explicitly specify channel to map the XZ plane')

parser.add_argument('-r', '--sub-rotation', type = int, 
                    help='Angle to rotation the sub-channel around the Y axis')

parser.add_argument('-k', '--keep-channels', action = 'store_true',
                    help='Process main and sub channels separately (useful for registration check).')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-e', '--reg-method', type = str, default = reg_method, choices = reg_method_list, \
                    help='Method used for image registration')

parser.add_argument('-t', '--opt-method', type = str, default = opt_method, choices = opt_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-i', '--iterations', type = int, default = iterations, \
                    help='number of iterations')

log.add_argument(parser, default_level = log_level)

parser.add_argument('input_file', default = input_filename, \
                    help='Input dual-view TIFF file')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
gpu_id = args.gpu_id
input_filename = args.input_file
main_channel = args.main_channel
sub_channel = args.sub_channel
sub_rotation = args.sub_rotation
keep_channels = args.keep_channels
reg_method = args.reg_method
opt_method = args.opt_method
psf_filename = args.psf_image
iterations = args.iterations

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
stack.turn_on_gpu(gpu_id)

# load input image
logger.info("Loading image: {0}.".format(input_filename))
input_stack = stack.Stack(input_filename)

if psf_filename is None:
    psf_path = Path(psf_folder).joinpath(psf_filename)
else:
    if Path(psf_filename).exists():
        psf_path = Path(psf_filename)
    else:
        psf_path = Path(psf_folder).joinpath(psf_filename)

logger.info("PSF image: {0}.".format(psf_path))
psf_stack = stack.Stack(psf_path)
psf_image = psf_stack.image_array[0, 0]

# finding the other channel
channel_set = set(np.arange(input_stack.c_count)) - {main_channel}
if sub_channel is None:
    sub_channel = min(channel_set)
logger.info("Selecting channels, main : {0}, sub: {1}.".format(main_channel, sub_channel))

# scaling along the z-axis to achieve isometric voxels
logger.info("Scaling image to be isometric. Pixel-size: {0}.".format(min(input_stack.voxel_um)))
input_stack.scale_isometric(gpu_id = gpu_id)
logger.debug("Image shaped into: {0}".format(input_stack.image_array.shape))

# rotate, register and deconvolve
output_stack = stack.Stack()
output_shape = list(input_stack.image_array.shape)
if keep_channels:
    output_shape[1] = 2
else:
    output_shape[1] = 1
output_stack.alloc_zero_image(output_shape, dtype = np.float)

affine_result_list = []
logger.info("Registration and deconvolution started.")
for index in progressbar(range(input_stack.t_count)):
    # registration
    main_image = input_stack.image_array[index, main_channel].astype(float)
    sub_image = input_stack.image_array[index, sub_channel].astype(float)

    if sub_rotation != 0:
        sub_image_rot = gpuimage.rotate_by_axis(sub_image, angle = sub_rotation, axis = 'y', gpu_id = gpu_id)
        sub_image_rot = gpuimage.resize(sub_image_rot, main_image.shape, centering = True)
    else:
        sub_image_rot = sub_image

    affine_result = register.register(main_image, sub_image_rot, gpu_id = gpu_id, \
                                      opt_method = opt_method, reg_method = reg_method)
    affine_result_list.append(affine_result)
    print(affine_result['matrix'])

    # deconvolution
    if iterations > 0:
        main_image = deconvolve.deconvolve(main_image, psf_image, iterations = iterations, gpu_id = gpu_id)
        sub_image = deconvolve.deconvolve(sub_image, psf_image, iterations = iterations, gpu_id = gpu_id)

    # affine transformation
    if sub_rotation != 0:
        sub_image_rot = gpuimage.rotate_by_axis(sub_image, angle = sub_rotation, axis = 'y', gpu_id = gpu_id)
        sub_image_rot = gpuimage.resize(sub_image_rot, main_image.shape, centering = True)
    else:
        sub_image_rot = sub_image

    sub_image = gpuimage.affine_transform(sub_image_rot, affine_result['matrix'], gpu_id = gpu_id)

    # fuse channels or just store them
    if keep_channels == False:
        output_stack.image_array[index, 0] = (main_image + sub_image) / 2
    else:
        output_stack.image_array[index, 0] = main_image
        output_stack.image_array[index, 1] = sub_image

# output in the ImageJ format, dimensions should be in TZCYX order
logger.info("Saving image: {0}.".format(output_filename))
output_stack.save_ome_tiff(output_filename, dtype = np.float32)
