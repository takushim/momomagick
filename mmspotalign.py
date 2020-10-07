#!/usr/bin/env python

import sys, argparse, pathlib, numpy, pandas, time
from statsmodels.nonparametric.smoothers_lowess import lowess

# defaults
input_filename = None
use_alignment = True
align_filename = 'align.txt'
use_smoothing = False
force_calc_smoothing = False
filename_suffix = '_aligned.txt'
shift_x = 0.0
shift_y = 0.0
scaling = 1.0
output_filename = None

parser = argparse.ArgumentParser(description='Align tracked spots.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--align-file', default = align_filename, \
                    help='a tsv file used for alignment')
parser.add_argument('-n', '--no-alignment', action='store_false', \
                   help='do not align spots')

parser.add_argument('-u', '--use-smoothing', action='store_true', default = use_smoothing, \
                   help='use previously calculated smoothing curves in the file')
parser.add_argument('-c', '--force-calc-smoothing', action='store_true', default = force_calc_smoothing, \
                   help='force (re)calculation to obtain smooth alignment curves')

parser.add_argument('-s', '--image-shift', nargs=2, type=float, default=[shift_x, shift_y], metavar=('X', 'Y'), \
                    help='shift of the image against spots')
parser.add_argument('-x', '--scaling', type = float, default = scaling, \
                    help='Scale cooredinates to use magnified images')

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input TSV file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
use_alignment = args.no_alignment
align_filename = args.align_file
use_smoothing = args.use_smoothing
force_calc_smoothing = args.force_calc_smoothing
shift_x, shift_y = args.image_shift
scaling = args.scaling

if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read TSV file
print("Read spots from {0}.".format(input_filename))
spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')

# saving original coordinates and parameters
spot_table['orig_x'] = spot_table['x']
spot_table['orig_y'] = spot_table['y']
spot_table['shift_x'] = shift_x
spot_table['shift_y'] = shift_y
spot_table['scaling'] = scaling

# shift spots
print("Shifting: ({0:f}, {1:f}), scaling: {2:f}".format(shift_x, shift_y, scaling))
spot_table['moved_x'] = spot_table['x'] * scaling + shift_x
spot_table['moved_y'] = spot_table['y'] * scaling + shift_y

# update coordinates
spot_table['x'] = spot_table['moved_x']
spot_table['y'] = spot_table['moved_y']

# alignment
if use_alignment:
    align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
    print("Using {0} for alignment.".format(align_filename))
    align_plane = numpy.array(align_table.align_plane)
    align_x = numpy.array(align_table.align_x)
    align_y = numpy.array(align_table.align_y)

    if use_smoothing:
        if ('smooth_x' not in align_table.columns) or ('smooth_y' not in align_table.columns) or \
        (force_calc_smoothing):
            print("Calculating smoothing. Smoothing data in the input file are ignored.")
            align_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
            align_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)
        else:
            print("Using smoothing data in the input file")
            align_x = numpy.array(align_table.smooth_x)
            align_y = numpy.array(align_table.smooth_y)

    # renew alignment table
    align_columns = ['align_plane', 'align_x', 'align_y']
    align_table = pandas.DataFrame({align_columns[0]: align_plane, \
                                    align_columns[1]: align_x, \
                                    align_columns[2]: align_y}, columns = align_columns)

    # alignment
    print("Alining using {0}.".format(align_filename))
    spot_table = pandas.merge(spot_table, align_table, \
                            left_on='plane', right_on='align_plane', how='left')
    spot_table['aligned_x'] = spot_table['moved_x']  - spot_table['align_x']
    spot_table['aligned_y'] = spot_table['moved_y']  - spot_table['align_y']

    # update coordinates
    spot_table['x'] = spot_table['aligned_x']
    spot_table['y'] = spot_table['aligned_y']

# output multipage tiff
print("Output TSV file to {0}.".format(output_filename))
spot_table.to_csv(output_filename, sep='\t', index=False)
