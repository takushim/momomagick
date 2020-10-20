#!/usr/bin/env python

import sys, pathlib, argparse, numpy, pandas
from mmtools import mmtiff, findarrow, lifetime, spotshift

# default values
input_filename = None
output_filename = None
filename_suffix = '_assigned.txt'
stereocilia_filename = 'Results.csv'
max_distance = None
spot_scaling = 1.0
spot_shift = [0.0, 0.0]
align_filename = None
use_smoothing = False
force_calc_smoothing = False
background_image_filename = None
separate_spots = False
output_image_filename = None
output_image_suffix = '_plotted.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Find nearest spots against each stereocilium', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output TSV file ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-r', '--stereocilia-file', default = stereocilia_filename, \
                    help='An ImageJ CSV file recording stereocilia arrows (from tips to roots)')
parser.add_argument('-m', '--max-distance', type = float, default = max_distance, \
                    help='maximum distance from stereocilia arrows (None = infinite)')

parser.add_argument('-x', '--spot-scaling', type = float, default = spot_scaling, \
                    help='scaling factor of detected spots')
parser.add_argument('-s', '--spot-shift', nargs = 2, type = float, default = spot_shift, \
                    metavar = ('shift_x', 'shift_y'), \
                    help='shift spots *after scaling*')

parser.add_argument('-a', '--align-file', default = align_filename, \
                    help='use a TSV file for spot alignment')
parser.add_argument('-u', '--use-smoothing', action='store_true', default = use_smoothing, \
                   help='use previously calculated smoothing curves in the file')
parser.add_argument('-c', '--force-calc-smoothing', action='store_true', default = force_calc_smoothing, \
                   help='force (re)calculation to obtain smooth alignment curves')

parser.add_argument('-b', '--background-image-file', default = background_image_filename, \
                    help='Draw results on a specified background image')
parser.add_argument('-e', '--separate-spots', action = 'store_true', default = separate_spots, \
                    help='separate the output image for each spot (debug mode)')
parser.add_argument('-g', '--output-image-file', default = output_image_filename, \
                    help='output TIFF file ([background_image]{0} by default)'.format(output_image_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='an input TSV file recodring detected spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

stereocilia_filename = args.stereocilia_file
max_distance = args.max_distance
spot_scaling = args.spot_scaling
spot_shift = args.spot_shift
align_filename = args.align_file
use_smoothing = args.use_smoothing
force_calc_smoothing = args.force_calc_smoothing
if args.background_image_file is not None:
    background_image_filename = args.background_image_file
    separate_spots = args.separate_spots
    if args.output_image_file is None:
        output_image_filename = mmtiff.MMTiff.stem(background_image_filename) + output_image_suffix
        if background_image_filename == output_image_filename:
            raise Exception('input_filename == output_filename.')
    else:
        output_image_filename = args.output_image_file

# load, shift and align spots
spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')
spot_shifter = spotshift.SpotShift(spot_scaling, spot_shift)
spot_table = spot_shifter.shift_spots(spot_table)
if align_filename is not None:
    print("Using {0} for alignment.".format(align_filename))
    align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
    if use_smoothing:
        if (not {'smooth_x', 'smooth_y'} <= set(align_table.columns)) or force_calc_smoothing:
            print("Calculating smoothing. Smoothing data in the input file are ignored.")
            align_table = spotshift.add_smoothing(align_table)
    spot_table = spotshift.SpotShift.align_spots(spot_table, align_table, use_smoothing, force_calc_smoothing)

# add lifetime
spot_table = lifetime.Lifetime.add_life_count(spot_table)

# keep first binding only
spot_table = spot_table.sort_values(['total_index', 'plane']).drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)

# assign stereocilia
stereocilia_finder = findarrow.FindArrow(stereocilia_filename)
stereocilia_finder.max_distance = max_distance
nearest_arrow_table = stereocilia_finder.find_nearest_arrow(spot_table)
spot_table = pandas.merge(spot_table, nearest_arrow_table, left_on='total_index', right_on='total_index', how='left')

# draw results
if background_image_filename is not None:
    # background image
    back_tiff = mmtiff.MMTiff(background_image_filename)
    back_image = back_tiff.as_array()[0, 0, 0]

    # add resolution culumns
    spot_table['arrow_pos_um'] = spot_table['arrow_pos'] * back_tiff.pixelsize_um

    if separate_spots:
        output_image = stereocilia_finder.draw_for_spot(back_image, spot_table)
    else:
        output_image = stereocilia_finder.draw_for_arrow(back_image, spot_table)

    # output ImageJ, dimensions should be in TZCYXS order
    print("Drawing results on", background_image_filename)
    back_tiff.save_image(output_image_filename, output_image)

# output TSV file
print("Output to a TSV file: {}".format(output_filename))
spot_table.to_csv(output_filename, sep='\t', index=False)

