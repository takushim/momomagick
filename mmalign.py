#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, itertools
from scipy.ndimage.interpolation import shift
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import mmtiff

# defaults
input_filename = None
align_filename = 'align.txt'
calc_smoothing = False
use_smoothing = False
filename_suffix = '_aligned.tif'
output_filename = None

parser = argparse.ArgumentParser(description='Align a multipage TIFF image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--align-file', default = align_filename, \
                    help='a tsv file used for alignment')

group = parser.add_mutually_exclusive_group()
group.add_argument('-c', '--calc-smoothing', action='store_true', default = calc_smoothing, \
                   help='smooth alignment curves by calculation')
group.add_argument('-u', '--use-smoothing', action='store_true', default = use_smoothing, \
                   help='use previously calculated smoothing curves in the file')

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
align_filename = args.align_file
calc_smoothing = args.calc_smoothing
use_smoothing = args.use_smoothing

if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
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
align_plane = numpy.array(align_table.align_plane)
align_x = numpy.array(align_table.align_x)
align_y = numpy.array(align_table.align_y)
if calc_smoothing:
    print("Calculating smoothing. Smoothing data in the input file are ignored.")
    align_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
    align_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)
elif use_smoothing:
    print("Using smoothing data in the input file")
    align_x = numpy.array(align_table.smooth_x)
    align_y = numpy.array(align_table.smooth_y)

move_x = move_x - align_x
move_y = move_y - align_y    

# align image
output_image = numpy.zeros_like(input_image)
for (time, zstack, channel) in itertools.product(range(input_tiff.total_time), range(input_tiff.total_zstack), range(input_tiff.total_channel)):
    output_image[time, zstack, channel] = shift(input_image[time, zstack, channel], (move_y[time], move_x[time]))
    print(time, zstack, channel, (move_x[time], move_y[time]))

# output multipage tiff
print("Output image file to %s." % (output_filename))
input_tiff.save_image(output_filename, output_image)
