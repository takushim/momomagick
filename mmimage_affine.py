#!/usr/bin/env python

import argparse
import numpy as np
from mmtools import stack, gpuimage, log, register

# defaults
input_filename = None
shift = [0.0, 0.0, 0.0]
rotation = [0.0, 0.0, 0.0]
zoom = [1.0, 1.0, 1.0]
shear = [0.0, 0.0, 0.0]
output_filename = None
output_suffix = '_affine.tif'
gpu_id = None

parser = argparse.ArgumentParser(description='Affine transformation of images.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-s', '--scale-isometric', action = 'store_true', \
                    help='Scale images to achieve isotrophic voxels')

parser.add_argument('-S', '--shift', nargs = 3, type = float, default = shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Shift of the image (useful with "-e None")')

parser.add_argument('-R', '--rotation', nargs = 3, type = float, default = rotation, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial rotation of the overlay image')

parser.add_argument('-Z', '--zoom', nargs = 3, type = float, default = zoom, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial zoom of the overlay image')

parser.add_argument('-H', '--shear', nargs = 3, type = float, default = shear, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial shear of the overlay image')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
scale_isometric = args.scale_isometric
if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

shift = np.array(args.shift[::-1])
rotation = np.array(args.rotation[::-1])
zoom = np.array(args.zoom[::-1])
shear = np.array(args.shear[::-1])

# activate GPU
if gpu_id is not None:
    gpuimage.turn_on_gpu(gpu_id)

# read input image
input_stack = stack.Stack(input_filename)
if scale_isometric:
    logger.info("Scaling to achieve isometric pixelsize: {0}.".format(min(input_stack.voxel_um)))
    input_stack.scale_isometric(gpu_id = gpu_id)

matrix = register.compose_matrix_3d(shift = shift, rotation = rotation, zoom = zoom, shear = shear)
logger.info("Affine matrix: {0}".format(matrix))
input_stack.affine_transform(matrix = matrix, gpu_id = gpu_id, progress = True)

# output TIFF
logger.info("Output image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename)
