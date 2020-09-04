#!/usr/bin/env python

import os, platform, sys, glob, argparse
import numpy, pandas, tifffile
from scipy.ndimage.interpolation import shift
from statsmodels.nonparametric.smoothers_lowess import lowess

# defaults
input_filename = None
align_filename = 'align.txt'
filename_suffix = '_aligned.tif'
output_filename = None

parser = argparse.ArgumentParser(description='Align fluorescent-spot image according to align.txt', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--align-filename', nargs=1, default = [align_filename], \
                    help='aligning tsv file name (align.txt if not specified)')

parser.add_argument('-o', '--output-file', nargs=1, default = None, \
                    help='output image file name ([basename]%s by default)' % (filename_suffix))

parser.add_argument('input_file', nargs=1, default=None, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
align_filename = args.align_filename[0]

if args.output_file is None:
    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file[0]

# read input image(s)
orig_images = tifffile.imread(input_filename)

# alignment
align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
print("Using %s for alignment." % (align_filename))
align_plane = numpy.array(align_table.align_plane)
align_x = numpy.array(align_table.align_x)
align_y = numpy.array(align_table.align_y)

align_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
align_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)

# align image
output_image = numpy.zeros_like(orig_images)
for index in range(len(orig_images)):
    align_index = numpy.where(align_plane == index)[0][0]
    shift_x = - align_x[align_index]
    shift_y = - align_y[align_index]
    print("Plane %d, Shift_X %f, Shift_Y %f" % (align_index, shift_x, shift_y))
    output_image[index] = shift(orig_images[index], [shift_y, shift_x], cval = 0)

# output multipage tiff
print("Output image file to %s." % (output_filename))
tifffile.imsave(output_filename, output_image)
