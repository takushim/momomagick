#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, tifffile
from scipy.ndimage.interpolation import shift

# default values
input_filename = None
output_filename = None
stamp_filename = 'TracEGFP_Decon.tif'
shift_x = 0
shift_y = 0
filename_suffix = '_stamp.tif'
xy_resolution = 0.1625
z_spacing = 0.5

# parse arguments
parser = argparse.ArgumentParser(description='Stamp one image to one stack', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-t', '--stamp-file', nargs=1, default=[stamp_filename], \
                    help='TIFF file to stamp')
parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='shift of the image to be stamped')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
stamp_filename = args.stamp_file[0]
shift_x, shift_y = args.image_shift
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF files
input_image = tifffile.imread(input_filename)
stamp_image = tifffile.imread(stamp_filename)
time, height, width = input_image.shape

# shift the stamp image
stamp_image = shift(stamp_image, (shift_y, shift_x))
print("Shifting the stamp image:", shift_x, shift_y)

# make an output image
output_image = numpy.zeros((time, 2, height, width), dtype = input_image.dtype)
for index in range(len(output_image)):
    output_image[index, 0] = stamp_image
    output_image[index, 1] = input_image[index]

# output ImageJ, dimensions should be in TZCYXS order
print('Output image was shaped into:', output_image.shape)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

