#!/usr/bin/env python

import sys, argparse, pathlib, numpy
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
    output_filename = pathlib.Path(input_filename).stem.split('.')[0] + '_out.tif'
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file[0]

# read TIFF file (assumes TZ(C)YX order)
orig_image = []
with tifffile.TiffFile(input_filename) as tiff:
    axes = tiff.series[0].axes
    image = tiff.asarray(series = 0)
    print('Found axis:', axes)
    print('Image shape', image.shape)
    if 'T' in axes:
        total_time = len(image)
        orig_image = numpy.split(image, len(image)) # becomes a list of [1, height, width]
    else:
        total_time = 1
        orig_image = [image]

if 'Z' not in axes:
    print('Adding temporary Z axis.')
    for index in range(len(orig_image)):
        orig_image[index] = orig_image[index][numpy.newaxis,]
if 'C' in axes:
    print('Using the first channel only.')
    for index in range(len(orig_image)):
        orig_image[index] = orig_image[index][..., 0:1, :, :]

orig_image = numpy.asarray(orig_image)
print('Original image was shaped into: ', orig_image.shape)

# overlay dimensions should be in TZCYXS order
total_frame = orig_image.shape[1]
if crop_height is None:
    crop_height = orig_image.shape[-2] - crop_y
if crop_width is None:
    crop_width = orig_image.shape[-1] // 2 - crop_x

output_image = numpy.zeros((total_time, total_frame, 2, crop_height, crop_width), dtype = orig_image.dtype)
print('Output image was shaped into: ', output_image.shape)
output_image[:, :, 1, :, :] = orig_image[:, :, 0, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]

split_shift = orig_image.shape[-1] // 2
shift_array = (0, 0, 0, shift_y, shift_x)
paste_image = orig_image[:, :, :, crop_y:(crop_y + crop_height), (split_shift + crop_x):(split_shift + crop_x + crop_width)]
paste_image = shift(paste_image, shift_array)
output_image[:, :, 0, :, :] = paste_image
print('Pasting image was shaped by', shift_array)

# output ImageJ, dimensions should be in TZCYXS order
#output_image = numpy.array(output_image).transpose(1, 0, 2, 3)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

