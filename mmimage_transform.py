#!/usr/bin/env python

import argparse
import numpy as np
from mmtools import stack, log, gpuimage

# default values
input_filename = None
output_filename = None
output_suffix = '_trans.tif'
gpu_id = None
crop_area = None
flip = ''
rot = [0.0, 0.0, 0.0]
shift = [0.0, 0.0, 0.0]

# parse arguments
parser = argparse.ArgumentParser(description='Rotate a tiff image.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='Output TIFF file ([basename0]{0} by default)'.format(output_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-s', '--scale-isometric', action = 'store_true', \
                    help='Scale images to achieve isotrophic voxels')

parser.add_argument('-C', '--crop-area', type = int, nargs = 4, default = crop_area, \
                    metavar = ('X', 'Y', 'W', "H"),
                    help='Crop using the specified area.')

parser.add_argument('-F', '--flip', type = str, default = flip, \
                    help='Flip the overlay image (e.g., X, XZ, XYZ)')

parser.add_argument('-R', '--rot', nargs = 3, type = float, default = rot, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Rotate the overlay image. Applied after flip. Consider isometric scaling.')

parser.add_argument('-S', '--shift', nargs = 3, type = float, default = shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Shift the overlay image. Applied after rotation.')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='Input dual-view TIFF file')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
crop_area = args.crop_area
scale_isometric = args.scale_isometric

flip = args.flip.lower()
rot = np.array(args.rot[::-1])
shift = np.array(args.shift[::-1])

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
gpuimage.turn_on_gpu(gpu_id)

# read input image
input_stack = stack.Stack(input_filename)
if scale_isometric:
    logger.info("Scaling to achieve isometric pixelsize: {0}.".format(min(input_stack.voxel_um)))
    input_stack.scale_isometric(gpu_id = gpu_id)

# cropping
if crop_area is not None:
    origin = [0] * len(input_stack.image_array.shape)
    shape = list(input_stack.image_array.shape)
    origin[3], origin[4] = crop_area[1], crop_area[0]   
    shape[3], shape[4] = crop_area[3], crop_area[2]

    logger.info("Cropping image. Origin: {0}. Shape: {1}".format(origin, shape))
    input_stack.crop_image(origin, shape)

# pre-process the first image
logger.info("Prosessing the image, flip: {0}, rotation: {1}, shift {2}.".format(flip, rot, shift))
flip_x, flip_y, flip_z = [-1 if axis in flip else 1 for axis in 'xyz']
def process (image, t_index, c_index):
    image = image[::flip_z, ::flip_y, ::flip_x]
    for axis, angle in zip(['x', 'y', 'z'], rot):
        if np.isclose(angle, 0.0) == False:
            image = gpuimage.rotate_by_axis(image, angle = angle, axis = axis, gpu_id = gpu_id)
    image = gpuimage.shift(image, shift, gpu_id = gpu_id)
    return image

input_stack.apply_all(process, progress = True)

# output TIFF
logger.info("Output image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename)
