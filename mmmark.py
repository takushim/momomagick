#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, tifffile
from mmtools import mmtiff, lifetime, drawspots

# prepare spot marker
spot_drawer = drawspots.DrawSpots()

# default values
input_filename = None
output_filename = None
marker_filename = 'spot_table.txt'
mask_filename = None
shift_x = 0.0
shift_y = 0.0
scaling = 1.0
filename_suffix = '_marked.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Mark detected spots on background images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output multipage TIFF file ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-s', '--image-shift', nargs=2, type=float, default=[shift_x, shift_y], metavar=('X', 'Y'), \
                    help='shift of the image against spots')
parser.add_argument('-x', '--scaling', type = float, default = scaling, \
                    help='Scale cooredinates to use magnified images')

parser.add_argument('-f', '--marker-file', default=marker_filename, \
                    help='name of TSV file (read [basename].txt if not specified)')
parser.add_argument('-z', '--marker-size', type=int, default=spot_drawer.marker_size, \
                    help='marker size to draw (dot == 0)')
parser.add_argument('-c', '--marker-colors', nargs=4, type=str, default=spot_drawer.marker_colors, \
                    metavar=('NEW', 'CONT', 'END', 'REDUN'), \
                    help='marker colors for new, tracked, disappearing, and redundant spots')
parser.add_argument('-r', '--rainbow-colors', action='store_true', default=spot_drawer.marker_rainbow, \
                    help='use rainbow colors to distinguish each tracking')

parser.add_argument('-m', '--mask-image', default = mask_filename, \
                    help='read masking image to omit unnecessary area')

parser.add_argument('-i', '--invert-image', action='store_true', default=spot_drawer.invert_image, \
                    help='invert the LUT of output image')

parser.add_argument('input_file', default=input_filename, \
                    help='input single or multi-page TIFF file to plot spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
mask_filename = args.mask_image
shift_x, shift_y = args.image_shift
scaling = args.scaling
marker_filename = args.marker_file
spot_drawer.marker_size = args.marker_size
spot_drawer.marker_colors = args.marker_colors
spot_drawer.marker_rainbow = args.rainbow_colors
spot_drawer.invert_image = args.invert_image

if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

# read TSV file
print("Read spots from {0}.".format(marker_filename))
spot_table = pandas.read_csv(marker_filename, comment = '#', sep = '\t')
total_planes = spot_table.plane.max() + 1

# shift spots
print("Shifting the stamp image:", shift_x, shift_y)
spot_table['x'] = spot_table['x'] * scaling + shift_x
spot_table['y'] = spot_table['y'] * scaling + shift_y

# read TIFF files TZCYX(S)
#input_image = tifffile.imread(input_filename)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()[:, 0, 0]

# filter spots with a masking image
if mask_filename is not None:
    mask_image = tifffile.imread(mask_filename)
    total_spots = len(spot_table)
    spot_table = lifetime.Lifetime.filter_spots_maskimage(spot_table, mask_image)
    print("Filtered {0:d} spots using a mask image: {1}.".format(total_spots - len(spot_table), mask_filename))

# make an output image
output_image = spot_drawer.convert_to_color(input_image)
if input_tiff.total_time == 1:
    output_image = numpy.array([output_image[0] for index in range(spot_table.plane.max())])

# make an output image
print("Marked {0:d} spots on {1}.".format(len(spot_table), input_filename))
output_image = spot_drawer.mark_spots(output_image, spot_table)

# output ImageJ, dimensions should be in TZCYXS order
print('Output image was shaped into:', output_image.shape)
input_tiff.save_image(output_filename, output_image)

