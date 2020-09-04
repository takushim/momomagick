#!/usr/bin/env python

import os, sys, platform, argparse, re, pathlib, numpy, itertools, tifffile
from mmtools import mmtiff
from scipy.ndimage.interpolation import shift
from PIL import Image, ImageDraw, ImageFont

# default values
input_filenames = None
output_filename = None
filename_suffix = '_over.tif'
#use_plane = 0
x_shift_range = [-20, 20, 1]
y_shift_range = [-20, 20, 1]
xy_resolution = 0.1625
z_spacing = 0.5

# font
if platform.system() == "Windows":
    font_file = 'C:/Windows/Fonts/Arial.ttf'
elif platform.system() == "Linux":
    font_file = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
elif platform.system() == "Darwin":
    font_file = '/Library/Fonts/Verdana.ttf'
else:
    raise Exception('font file error.')
font_size = 20
font_color = 'white'
font = ImageFont.truetype(font_file, font_size)

# parse arguments
parser = argparse.ArgumentParser(description='Try overlay of two images using various alignments', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='filename of output TIFF file ([basename]%s by default)' % (filename_suffix))

#parser.add_argument('-p', '--use-plane', nargs=1, type=int, default=[use_plane], \
#                    help='frame to detect spots (the first frame if not specified)')

parser.add_argument('-X', '--x-shift-range', nargs=3, type=float, default = x_shift_range, \
                    metavar=('BEGIN', 'END', 'STEP'), \
                    help='range of x shift to try for Image 0 (accepting floats)')

parser.add_argument('-Y', '--y-shift-range', nargs=3, type=float, default = y_shift_range, \
                    metavar=('BEGIN', 'END', 'STEP'), \
                    help='range of y shift to try for Image 0 (accepting floats)')

parser.add_argument('input_file', nargs=2, default=input_filenames, \
                    help='two input (multipage) TIFF files (image_0, image_1)')
args = parser.parse_args()

# set arguments
input_filenames = args.input_file
#use_plane = args.use_plane[0]
x_shift_range = args.x_shift_range
y_shift_range = args.y_shift_range
if args.output_file is None:
    stem = pathlib.Path(input_filenames[0]).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('_[0-9]+$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
else:
    output_filename = args.output_file[0]

# read TIFF file (assumes TZCYX order)
input_tiffs = []
input_tiffs.append(mmtiff.MMTiff(input_filenames[0]))
input_tiffs.append(mmtiff.MMTiff(input_filenames[1]))

input_images = []
input_images.append(input_tiffs[0].as_array()[0, 0, 0])
input_images.append(input_tiffs[1].as_array()[0, 0, 0])

overlay_width = max([x.width for x in input_tiffs])
overlay_height = max([x.height for x in input_tiffs])
output_dtype = numpy.uint16

shift_xs = numpy.arange(*x_shift_range)
shift_ys = numpy.arange(*y_shift_range)
output_images = []
for (shift_x, shift_y) in itertools.product(shift_xs, shift_ys):
    # prepare output image space
    overlay_images = []
    shifted_image = numpy.zeros((overlay_height + font_size, overlay_width), dtype = output_dtype)
    shifted_image[0:overlay_height, :] = shift(input_images[0], (shift_y, shift_x))
    overlay_images.append(shifted_image)

    orig_image = numpy.zeros((overlay_height + font_size, overlay_width), dtype = output_dtype)
    orig_image[0:overlay_height, :] = input_images[1]
    overlay_images.append(orig_image)

    # draw condition
    font_image = numpy.zeros((font_size, overlay_width), dtype = output_dtype)
    image = Image.fromarray(font_image)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "X %.1f Y %.1f" % (shift_x, shift_y), font = font, fill = font_color)
    font_image = numpy.array(image)
    if font_image.dtype is numpy.uint16:
        font_image = (font_image << 8)
    for index in range(len(overlay_images)):
        overlay_images[index][overlay_height:, :] = numpy.array(font_image)

    print("Done for", shift_x, shift_y)
    output_images.append(numpy.array(overlay_images))

# output ImageJ, dimensions should be in TZCYXS order
tifffile.imsave(output_filename, numpy.array(output_images), imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})
print("Output image:", output_filename)
