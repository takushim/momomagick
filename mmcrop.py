#!/usr/bin/env python

import sys, argparse, pathlib, numpy
from skimage.external import tifffile
from scipy.ndimage.interpolation import shift
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
crop_x = 0
crop_y = 0
crop_width = None
crop_height = None
use_channel = None
filename_suffix = '_crop.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Crop diSPIM image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-g', '--image-origin', nargs=2, type=int, default=[crop_x, crop_y], \
                    metavar=('X', 'Y'), \
                    help='origin of cropping')
parser.add_argument('-z', '--image-size', nargs=2, type=int, default=[crop_width, crop_height], \
                    metavar=('WIDTH', 'HEIGHT'), \
                    help='size of cropped image')
parser.add_argument('-c', '--use-channel', nargs=1, type=int, default=use_channel, \
                    help='channel to output (None = all)')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
crop_x, crop_y = args.image_origin
crop_width, crop_height = args.image_size
if args.use_channel is not None:
    use_channel = args.use_channel[0]
if args.output_file is None:
    output_filename = pathlib.Path(input_filename).stem.split('.')[0] + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF file (assumes TZ(C)YX order)
image_file = mmtiff.MMTiff(input_filename)
#if image_file.colored:
#    raise Exception('Color image not accepted.')

# remove unnecessary channel(s)
orig_image = image_file.as_array(channel = use_channel, drop_channel = False)
output_image = orig_image[:, :, :, crop_y:(crop_y + crop_height), crop_x:(crop_x + crop_width)]

# output ImageJ, dimensions should be in TZCYXS order
xy_resolution = image_file.pixelsize_um
z_spacing = image_file.z_step_um
print('Output image was shaped into: ',output_image.shape)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

