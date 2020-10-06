#!/usr/bin/env python

import os, sys, argparse, pathlib, numpy, tifffile
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
crop_x = 1410
crop_y = 0
crop_width = 256
crop_height = 256
#back_value = 103
back_value = 0
stack_range = None
use_channel = 0
dirname_spima = 'SPIMA'
dirname_spimb = 'SPIMB'
#dirname_output = 'Output'
filename_spima = 'SPIMA_0.tif'
filename_spimb = 'SPIMB_0.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Crop and save diSPIM image for deconvolution', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--image-origin', nargs=2, type=int, default = [crop_x, crop_y], \
                    metavar=('X', 'Y'), \
                    help='origin of cropping')
parser.add_argument('-z', '--image-size', nargs=2, type=int, default = [crop_width, crop_height], \
                    metavar=('WIDTH', 'HEIGHT'), \
                    help='size of cropped image')
parser.add_argument('-s', '--stack-range', nargs=2, type=int, default = stack_range, \
                    metavar=('BEGIN', 'END'), \
                    help='range of stack to be used')
parser.add_argument('-b', '--back-value', type=int, default = back_value, \
                    help='background_value to subtract')
parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
crop_x, crop_y = args.image_origin
crop_width, crop_height = args.image_size
back_value = args.back_value
stack_range = args.stack_range

# read TIFF file (assumes TZ(C)YX order)
image_file = mmtiff.MMTiff(input_filename)

# remove unnecessary channel(s)
orig_image = image_file.as_array()
orig_image = orig_image[:, :, :, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]
print('Input image was shaped into: ', orig_image.shape)
#total_frame = orig_image.shape[1]
#half_size = int(total_frame / 2)

# prepare output images
if stack_range is None:
    spima_image = orig_image[:, :, 0:1].copy()
    spimb_image = orig_image[:, :, 1:2].copy()
else:
    spima_image = orig_image[:, stack_range[0]:(stack_range[1] + 1), 0:1].copy()
    spimb_image = orig_image[:, stack_range[0]:(stack_range[1] + 1), 1:2].copy()

# subtract a background value
spima_image = numpy.clip(spima_image, back_value, None) - back_value
spimb_image = numpy.clip(spimb_image, back_value, None) - back_value

# output ImageJ, dimensions should be in TZCYXS order
os.makedirs(dirname_spima, exist_ok = True)
os.makedirs(dirname_spimb, exist_ok = True)
#os.makedirs(dirname_output, exist_ok = True)
xy_resolution = image_file.pixelsize_um
z_spacing = image_file.z_step_um
print('SPIMA image was shaped into: ', spima_image.shape)
tifffile.imsave(os.path.join(dirname_spima, filename_spima), spima_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})
print('SPIMB image was shaped into: ', spima_image.shape)
tifffile.imsave(os.path.join(dirname_spimb, filename_spimb), spimb_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})
