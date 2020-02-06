#!/usr/bin/env python

import os, sys, glob, argparse, numpy
from skimage.external import tifffile
from scipy.ndimage.interpolation import shift

# default values
input_filename = None
output_filename = None
crop_x = 0
crop_y = 0
crop_width = None
crop_height = None
shift_x = 0
shift_y = 0
xy_resolution = 1 / 6.1538 # um per pixel (6.1538 is the scale in ImageJ)
z_spacing = 0.5 # um per plane

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two-channel split diSPIM image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]_out.tif by default)')
parser.add_argument('-g', '--image-origin', nargs=2, type=int, default=[crop_x, crop_y], \
                    metavar=('X', 'Y'), \
                    help='origin of image used to overlay')
parser.add_argument('-z', '--image-size', nargs=2, type=int, default=[crop_width, crop_height], \
                    metavar=('WIDTH', 'HEIGHT'), \
                    help='size of image used to overlay')
parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='ajustment for overlay')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
crop_x, crop_y = args.image_origin
crop_width, crop_height = args.image_size
shift_x, shift_y = args.image_shift
if args.output_file is None:
    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + '_out.tif'
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file[0]

# read TIFF file
orig_image = tifffile.imread(input_filename)
if len(orig_image.shape) == 2:
    orig_image = numpy.array([orig_image])

# overlay
total_frame = orig_image.shape[-3]
if crop_width is None:
    crop_width = orig_image.shape[-1] // 2 - crop_x
if crop_height is None:
    crop_height = orig_image.shape[-2] - crop_y

output_image = numpy.zeros((2, total_frame, crop_height, crop_width), dtype = orig_image.dtype)
output_image[1] = orig_image[:, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]

print(orig_image.shape, output_image.shape)
split_shift = orig_image.shape[-1] // 2
paste_image = orig_image[:, crop_y:(crop_y + crop_height), (split_shift + crop_x):(split_shift + crop_x + crop_width)]
print(paste_image.shape)
output_image[0] = shift(paste_image, (0, shift_y, shift_x))

# output ImageJ, dimensions should be in TZCYXS order
output_image = numpy.array(output_image).transpose(1, 0, 2, 3)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

