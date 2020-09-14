#!/usr/bin/env python

import sys, pathlib, argparse, numpy, pandas
from mmtools import mmtiff, findarrow

# default values
input_filename = None
output_filename = None
spot_scaling = 1.0
spot_shift = [0.0, 0.0]
filename_suffix = '_assigned.txt'
stereocilia_filename = 'Results.csv'
background_image_filename = None
draw_for_spot = False
output_image_filename = None
output_image_suffix = '_plotted.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Find nearest spots against each stereocilium', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output TSV file ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-t', '--stereocilia-file', default = stereocilia_filename, \
                    help='An ImageJ CSV file recording stereocilia arrows (from tips to roots)')

parser.add_argument('-x', '--spot-scaling', type = float, default = spot_scaling, \
                    help='scaling factor of detected spots')
parser.add_argument('-s', '--spot-shift', nargs = 2, type = float, default = spot_shift, \
                    metavar = ('shift_x', 'shift_y'), help='shift of markers')

parser.add_argument('-b', '--background-image-file', default = background_image_filename, \
                    help='a background image to draw the results')
parser.add_argument('-w', '--draw-for-spot', action = 'store_true', default = draw_for_spot, \
                    help='draw images for each spot (debug mode)')
parser.add_argument('-i', '--output-image-file', default = output_image_filename, \
                    help='output TIFF file ([background_image]{0} by default)'.format(output_image_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='an input TSV file recodring detected spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

spot_scaling = args.spot_scaling
spot_shift = args.spot_shift
stereocilia_filename = args.stereocilia_file
background_image_filename = args.background_image_file
draw_for_spot = args.draw_for_spot
if args.background_image_file is not None:
    if args.output_image_file is None:
        output_image_filename = mmtiff.MMTiff.filename_stem(background_image_filename) + output_image_suffix
        if background_image_filename == output_image_filename:
            raise Exception('input_filename == output_filename.')
    else:
        output_image_filename = args.output_image_file

# load spots
spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')
spot_table = spot_table.sort_values(['total_index', 'plane']).drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)

# assign stereocilia
stereocilia_finder = findarrow.FindArrow(stereocilia_filename, spot_shift, spot_scaling)
nearest_arrow_table = stereocilia_finder.find_nearest_arrow(spot_table)
spot_table = pandas.merge(spot_table, nearest_arrow_table, left_on='total_index', right_on='total_index', how='left')

# output TSV file
output_file = open(output_filename, 'w', newline='')
stereocilia_finder.output_header(output_file)
spot_table.to_csv(output_file, sep='\t', index=False, header=True, mode='a')
output_file.close()

# draw results
if background_image_filename is not None:
    # background image
    back_tiff = mmtiff.MMTiff(background_image_filename)
    back_image = back_tiff.as_array()[0, 0, 0]

    if draw_for_spot:
        output_image = stereocilia_finder.draw_for_spot(back_image, spot_table)
    else:
        output_image = stereocilia_finder.draw_for_arrow(back_image, spot_table)

    # output ImageJ, dimensions should be in TZCYXS order
    print("Draw results on", background_image_filename)
    back_tiff.save_image(output_image_filename, output_image)
