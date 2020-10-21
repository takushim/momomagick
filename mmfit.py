#!/usr/bin/env python

import os, sys, argparse, re, pathlib, numpy, itertools
from mmtools import mmtiff
from scipy.ndimage.interpolation import shift
from PIL import Image, ImageDraw, ImageFont

# default values
input_filenames = None
output_filename = None
filename_suffix = '_fit.tif'
use_plane = 0
shift_range_x = [-20, 20, 0.5]
shift_range_y = [-10, 10, 0.5]

# font
font_file = mmtiff.MMTiff.font_path()
#font_color = 'white'
font_color = 128

# parse arguments
parser = argparse.ArgumentParser(description='Try overlay of two images using various alignments', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='filename of output TIFF file ([basename]%s by default)' % (filename_suffix))

parser.add_argument('-p', '--use-plane', type=int, default=use_plane, \
                    help='frame to detect spots (the first frame if not specified)')

group = parser.add_mutually_exclusive_group()
group.add_argument('-X', '--shift-range-x', nargs=3, type=float, default = shift_range_x, \
                   metavar=('BEGIN', 'END', 'STEP'), \
                   help='range of x shift to try for Image 0 (accepting floats)')
group.add_argument('-x', '--shift-fix-x', type = float, default = None, \
                   help='fix x shift at a specified value')

group = parser.add_mutually_exclusive_group()
group.add_argument('-Y', '--shift-range-y', nargs=3, type=float, default = shift_range_y, \
                   metavar=('BEGIN', 'END', 'STEP'), \
                   help='range of y shift to try for Image 0 (accepting floats)')
group.add_argument('-y', '--shift-fix-y', type = float, default = None, \
                   help='fix y shift at a specified value')

parser.add_argument('input_file', nargs=2, default=input_filenames, \
                    help='two input (multipage) TIFF files (overlay, background)')
args = parser.parse_args()

# set arguments
input_filenames = args.input_file
use_plane = args.use_plane

if args.shift_fix_x is None:
    shift_range_x = args.shift_range_x
else:
    shift_range_x = [args.shift_fix_x, args.shift_fix_x, 1.0]

if args.shift_fix_y is None:
    shift_range_y = args.shift_range_y
else:
    shift_range_y = [args.shift_fix_y, args.shift_fix_y, 1.0]

if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filenames[0]) + filename_suffix
else:
    output_filename = args.output_file

# read TIFF file (assumes TZCYX order)
input_tiffs = []
input_tiffs.append(mmtiff.MMTiff(input_filenames[0]))
input_tiffs.append(mmtiff.MMTiff(input_filenames[1]))

input_images = []
input_images.append(input_tiffs[0].as_array()[use_plane, 0, 0])
input_images.append(input_tiffs[1].as_array()[0, 0, 0])
if use_plane > 0:
    print("Using plane {0} of original image.".format(use_plane))

overlay_width = max([x.width for x in input_tiffs])
overlay_height = max([x.height for x in input_tiffs])
output_dtype = numpy.uint16

font_size = overlay_height // 8
font = ImageFont.truetype(font_file, font_size)

print("X range", shift_range_x)
print("Y range", shift_range_y)
shift_xs = numpy.arange(shift_range_x[0], shift_range_x[1] + shift_range_x[2], shift_range_x[2])
shift_ys = numpy.arange(shift_range_y[0], shift_range_y[1] + shift_range_y[2], shift_range_y[2])
output_images = []
for (shift_y, shift_x) in itertools.product(shift_ys, shift_xs):
    # prepare output image space
    overlay_images = []

    # background
    orig_image = numpy.zeros((overlay_height, overlay_width), dtype = output_dtype)
    orig_image[0:input_tiffs[1].height, 0:input_tiffs[1].width] = input_images[1].copy()

    # draw shifts
    #font_color = numpy.max(orig_image)
    image = Image.fromarray(orig_image)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "X %+04.1f Y %+04.1f" % (shift_x, shift_y), font = font, fill = font_color)
    orig_image = numpy.array(image)
    overlay_images.append(orig_image)

    # overlay
    shifted_image = numpy.zeros((overlay_height, overlay_width), dtype = output_dtype)
    shifted_image[0:input_tiffs[0].height, 0:input_tiffs[0].width] = input_images[0]
    overlay_images.append(shift(shifted_image, (shift_y, shift_x)))

    output_images.append(numpy.array(overlay_images))

# output ImageJ, dimensions should be in TZCYXS order
print("Output image:", output_filename)
input_tiffs[0].save_image(output_filename, numpy.array(output_images))
