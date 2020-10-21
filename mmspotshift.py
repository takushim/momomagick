#!/usr/bin/env python

import sys, argparse, pathlib, numpy, pandas, time
from mmtools import spotshift

# defaults
input_filename = None
filename_suffix = '_shifted.txt'
spot_scaling = 1.0
spot_shift = [0.0, 0.0]
align_filename = None
use_smoothing = False
force_calc_smoothing = False
output_filename = None

parser = argparse.ArgumentParser(description='Align tracked spots.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--spot-scaling', type = float, default = spot_scaling, \
                    help='Scale cooredinates to use magnified images')
parser.add_argument('-s', '--spot-shift', nargs=2, type=float, default=spot_shift, metavar=('X', 'Y'), \
                    help='Shift of the image against spots *after scaling*')

parser.add_argument('-a', '--align-file', default = align_filename, \
                    help='a tsv file used for alignment')
parser.add_argument('-u', '--use-smoothing', action='store_true', default = use_smoothing, \
                   help='use previously calculated smoothing curves in the file')
parser.add_argument('-c', '--force-calc-smoothing', action='store_true', default = force_calc_smoothing, \
                   help='force (re)calculation to obtain smooth alignment curves')

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input TSV file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
spot_scaling = args.spot_scaling
spot_shift = args.spot_shift
align_filename = args.align_file
use_smoothing = args.use_smoothing
force_calc_smoothing = args.force_calc_smoothing

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

# shift spots
spot_shifter = spotshift.SpotShift(spot_scaling, spot_shift)
spot_table = spot_shifter.shift_spots(spot_table)

# alignment
if align_filename is not None:
    align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
    print("Using {0} for alignment.".format(align_filename))
    if use_smoothing:
        if (not {'smooth_x', 'smooth_y'} <= set(align_table.columns)) or force_calc_smoothing:
            print("Calculating smoothing. Smoothing data in the input file are ignored.")
            align_table = spotshift.add_smoothing(align_table)
    spot_table = spotshift.SpotShift.align_spots(spot_table, align_table, use_smoothing, force_calc_smoothing)

# output multipage tiff
print("Output TSV file to {0}.".format(output_filename))
spot_table.to_csv(output_filename, sep='\t', index=False)
