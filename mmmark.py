#!/usr/bin/env python

import argparse, json
from mmtools import stack, trackj, particles, log

# default values
input_filename = None
output_filename = None
output_suffix = '_marked.tif'
record_filename = None
record_suffix = '_track.json'
marker_size = 4
marker_colors = ['red', 'orange', 'blue']
rainbow_list = ["red", "blue", "green", "magenta", "purple", "cyan", "orange", "maroon"]

# parse arguments
parser = argparse.ArgumentParser(description='Mark detected spots on background images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file ([basename]{0} by default)'.format(output_suffix))

parser.add_argument('-f', '--record-file', default = record_filename, \
                    help='TSV file or TrackJ CSV file ([basename].txt if not specified)')

parser.add_argument('-z', '--marker-size', type = int, default = marker_size, \
                    help='marker size to draw')

parser.add_argument('-c', '--marker-colors', nargs = 3, type=str, metavar=('NEW', 'CONT', 'END'), \
                    help='marker colors for new, tracked, disappearing, and redundant spots')

parser.add_argument('-r', '--marker-rainbow', action = 'store_true', \
                    help='use rainbow colors to distinguish each tracking')

parser.add_argument('-i', '--invert-lut', action = 'store_true', \
                    help='invert the LUT of output image')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input image file.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
marker_size = args.marker_size
marker_colors = args.marker_colors
marker_rainbow = args.marker_rainbow
invert_lut = args.invert_lut

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

if args.record_file is None:
    record_filename = stack.with_suffix(input_filename, record_suffix)
    log.info("")
else:
    record_filename = args.record_file



# read TSV or TrackJ CSV file



if pathlib.Path(marker_filename).suffix.lower() == ".txt":
    print("Read TSV from {0}.".format(marker_filename))
    spot_table = pandas.read_csv(marker_filename, comment = '#', sep = '\t')
elif pathlib.Path(marker_filename).suffix.lower() == ".csv":
    print("Read TrackJ CSV from {0}.".format(marker_filename))
    spot_table = trackj.TrackJ(marker_filename).spot_table
else:
    raise Exception("Unknown file format.")
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
    spot_table = lifetime.filter_spots_maskimage(spot_table, mask_image)
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

