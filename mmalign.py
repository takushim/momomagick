#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, itertools
from scipy.ndimage.interpolation import shift
from mmtools import mmtiff, spotshift

# defaults
input_filename = None
align_filename = 'align.txt'
use_smoothing = False
force_calc_smoothing = False
filename_suffix = '_aligned.tif'
output_filename = None

parser = argparse.ArgumentParser(description='Align a multipage TIFF image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--align-file', default = align_filename, \
                    help='a tsv file used for alignment')

parser.add_argument('-u', '--use-smoothing', action='store_true', default = use_smoothing, \
                   help='use previously calculated smoothing curves in the file')
parser.add_argument('-c', '--force-calc-smoothing', action='store_true', default = force_calc_smoothing, \
                   help='force (re)calculation to obtain smooth alignment curves')

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
align_filename = args.align_file
use_smoothing = args.use_smoothing
force_calc_smoothing = args.force_calc_smoothing

if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read input image(s)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

# alignment
move_x = numpy.zeros(input_tiff.total_time)
move_y = numpy.zeros(input_tiff.total_time)

align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
print("Using %s for alignment." % (align_filename))
if use_smoothing:
    if (not {'smooth_x', 'smooth_y'} <= set(align_table.columns)) or force_calc_smoothing:
        print("Calculating smoothing. Smoothing data in the input file are ignored.")
        align_table = spotshift.SpotShift.add_smoothing(align_table)
    move_x = move_x - numpy.array(align_table.smooth_x)
    move_y = move_y - numpy.array(align_table.smooth_y)
else:
    move_x = move_x - numpy.array(align_table.align_x)
    move_y = move_y - numpy.array(align_table.align_y)

# align image
output_image = numpy.zeros_like(input_image)
for (time, zstack, channel) in itertools.product(range(input_tiff.total_time), range(input_tiff.total_zstack), range(input_tiff.total_channel)):
    output_image[time, zstack, channel] = shift(input_image[time, zstack, channel], (move_y[time], move_x[time]))
    #print(time, zstack, channel, (move_x[time], move_y[time]))
print("Final movements: {0}, {1}".format(move_x[-1], move_y[-1]))

# output multipage tiff
print("Output image file to %s." % (output_filename))
input_tiff.save_image(output_filename, output_image)
