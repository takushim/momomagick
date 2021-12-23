#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff, gpuimage, register

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

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
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
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')
input_images = input_tiff.as_list(list_channel = True)

output_image_list = []
pixelsize_um = input_tiff.pixelsize_um
z_step_um = input_tiff.z_step_um
z_ratio = z_step_um / pixelsize_um
print("Z-zoom ratio:", z_ratio)

matrix = register.compose_matrix_3d(shift = shift, rotation = rotation, zoom = zoom, shear = shear)
print("Matrix:")
print(matrix)

print("Transforming image:", end = ' ')
for index in range(input_tiff.total_time):
    print(index, end = ' ', flush = True)

    channel_image_list = []
    for channel in range(input_tiff.total_channel):
        input_image = input_images[index][channel].astype(float)
        if np.isclose(z_ratio, 1.0) == False:
            input_image = gpuimage.z_zoom(input_image, ratio = z_ratio, gpu_id = gpu_id)
        output_image = gpuimage.affine_transform(input_image, matrix, gpu_id = gpu_id)
        channel_image_list.append(output_image)
    output_image_list.append(channel_image_list)
print(".")

# output image
print("Output image:", output_filename)
output_image = np.array(output_image_list).swapaxes(1, 2).astype(np.float32)
mmtiff.save_image(output_filename, output_image, imagej = True, \
                  xy_pixel_um = pixelsize_um, \
                  z_step_um = z_step_um, \
                  finterval_sec = input_tiff.finterval_sec)
