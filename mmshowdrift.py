#!/usr/bin/env python

import os, platform, sys, glob, argparse, numpy, pandas
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess

# defaults
input_filename = 'align.txt'
output_filename = 'align.png'

parser = argparse.ArgumentParser(description='Show a graph of alignment curves', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name')
parser.add_argument('input_file', nargs = '?', default=input_filename, \
                    help='input a TSV file to prepare a graph')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
output_filename = args.output_file

# prepare a graph
align_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')

align_plane = numpy.array(align_table.align_plane)
align_x = numpy.array(align_table.align_x)
align_y = numpy.array(align_table.align_y)

smooth_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
smooth_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)

# draw a graph
pyplot.plot(align_plane, align_x)
pyplot.plot(align_plane, align_y)
pyplot.plot(align_plane, smooth_x)
pyplot.plot(align_plane, smooth_y)
pyplot.savefig(output_filename)
pyplot.show()


