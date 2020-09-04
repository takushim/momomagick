#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, tifffile
from scipy.ndimage.interpolation import shift
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
crop_x = 0
crop_y = 0
crop_width = None
crop_height = None
shift_x = 0
shift_y = 0
use_channel = 0
filename_suffix = '_over.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two-channel split diSPIM image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-g', '--image-origin', nargs=2, type=int, default=[crop_x, crop_y], \
                    metavar=('X', 'Y'), \
                    help='origin of image used to overlay')
parser.add_argument('-z', '--image-size', nargs=2, type=int, default=[crop_width, crop_height], \
                    metavar=('WIDTH', 'HEIGHT'), \
                    help='size of image used to overlay')
parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='ajustment for overlay')
parser.add_argument('-c', '--use-channel', nargs=1, type=int, default=[use_channel], \
                    help='channel to output')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
crop_x, crop_y = args.image_origin
crop_width, crop_height = args.image_size
shift_x, shift_y = args.image_shift
use_channel = args.use_channel[0]
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF file (assumes TZ(C)YX order)
image_file = mmtiff.MMTiff(input_filename)
if image_file.colored:
    raise Exception('Color image not accepted.')

# remove unnecessary channel(s)
orig_image = image_file.as_list(channel = use_channel, drop_channel = True)

# overlay dimensions should be in TZCYXS order
total_zstack = image_file.total_zstack
total_width = image_file.width
split_shift = total_width // 2
if crop_height is None:
    crop_height = image_file.height - crop_y
if crop_width is None:
    crop_width = total_width // 2 - crop_x
shift_array = (0, shift_y, shift_x)

output_image = []
for index in range(len(orig_image)):
    image = numpy.zeros((total_zstack, 2, crop_height, crop_width), dtype = image_file.dtype)
    image[:, 1, :, :] = orig_image[index][:, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]

    paste_image = shift(orig_image[index][:, :, split_shift:total_width], shift_array)
    image[:, 0, :, :] = paste_image[:, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]

    output_image.append(image)
    print('Index:', index, 'Shape:', image.shape, 'Shift:', shift_array)

# output ImageJ, dimensions should be in TZCYXS order
output_image = numpy.array(output_image)
xy_resolution = image_file.pixelsize_um
z_spacing = image_file.z_step_um
print('Output image was shaped into:', output_image.shape)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

