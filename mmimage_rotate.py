#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff, gpuimage

# default values
input_filename = None
output_filename = None
output_suffix = '_rot.tif'
gpu_id = None
crop_area = None
use_channel = None
rotation_angles = [0.0, 0.0, 0.0]

# parse arguments
parser = argparse.ArgumentParser(description='Rotate a tiff image.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='Output TIFF file ([basename0]{0} by default)'.format(output_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-a', '--rotation-angles', type = float, nargs = 3, default = rotation_angles, \
                    metavar = ('X', 'Y', 'Z'),
                    help='Rotation angles.')

parser.add_argument('-R', '--crop-area', type = int, nargs = 4, default = crop_area, \
                    metavar = ('X', 'Y', 'W', "H"),
                    help='Crop using the specified area.')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='Specify channel to output (None for all)')

parser.add_argument('input_file', default = input_filename, \
                    help='Input dual-view TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
rotation_angles = args.rotation_angles
crop_area = args.crop_area
use_channel = args.use_channel

if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
gpuimage.turn_on_gpu(gpu_id)

# read input TIFF
input_tiff = mmtiff.MMTiff(input_filename)
input_image_list = input_tiff.as_list(list_channel = True)

# scaling along the z-axis to achieve isometric voxels
z_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
print("Z scaling ratio:", z_ratio)

# cropping
if crop_area is None:
    print("Using the entire image.")
    crop_area = [0, 0, input_tiff.width, input_tiff.height]
else:
    print("Cropping using area:", crop_area)
x_slice, y_slice = mmtiff.area_to_slice(crop_area)

if use_channel is None:
    c_range = range(0, input_tiff.total_channel)
else:
    c_range = range(use_channel, use_channel + 1)

# rotate images
output_image_list = []
print("Rotation (X-Y-Z):", rotation_angles)
for index in range(input_tiff.total_time):
    channel_list = []
    for channel in c_range:
        image = input_image_list[index][channel][:, y_slice, x_slice].astype(float)
        image = gpuimage.z_zoom(image, ratio = z_ratio, gpu_id = gpu_id)

        for axis, angle in zip(['x', 'y', 'z'], rotation_angles):
            if np.isclose(angle, 0.0) == False:
                image = gpuimage.rotate(image, angle = angle, axis = axis, gpu_id = gpu_id)

        channel_list.append(image)
        
    # store the image to the list
    output_image_list.append(channel_list)

# output in the ImageJ format, dimensions should be in TZCYX order
print("Output image:", output_filename)
output_image = np.array(output_image_list).swapaxes(1, 2).astype(np.float32)
mmtiff.save_image(output_filename, output_image, \
                  xy_res = 1 / input_tiff.pixelsize_um, z_step_um = input_tiff.pixelsize_um, 
                  finterval = input_tiff.finterval_sec)
