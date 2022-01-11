#!/usr/bin/env python

import argparse, logging
from mmtools import stack

# default values
input_filename = None
output_filename = None
output_suffix = "_crop.tif"
channel = None
reverse_channel = False
crop_area = None
z_range = None
t_range = None
log_level = 'INFO'

# parse arguments
parser = argparse.ArgumentParser(description='Crop a multi-page TIFF image.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix))

parser.add_argument('-r', '--reverse-channel', action = 'store_true', \
                    help='reverse the order of channels before selecting channel(s).')

parser.add_argument('-c', '--channel', type = int, default = channel, \
                    help='specify the channel to process (minus index to remove).')

parser.add_argument('-R', '--crop-area', type = int, nargs = 4, default = crop_area, \
                    metavar = ('X', 'Y', 'W', "H"),
                    help='Crop using the specified area.')

parser.add_argument('-z', '--z-range', type = int, nargs = 2, default = z_range, \
                    metavar = ('START', 'END'),
                    help='Specify the range of z planes to output')

parser.add_argument('-t', '--t-range', type = int, nargs = 2, default = t_range, \
                    metavar = ('START', 'END'),
                    help='Specify the range of time frames to output')

parser.add_argument('-L', '--log-level', default = log_level, \
                    help='Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL')

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# logging
log_level = args.log_level
logging.basicConfig(format = '%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(log_level)

# set arguments
input_filename = args.input_file
channel = args.channel
reverse_channel = args.reverse_channel
z_range = args.z_range
t_range = args.t_range
crop_area = args.crop_area

if args.output_file is None:
    output_filename.append(stack.with_suffix(input_filename, output_suffix))
else:
    output_filename = args.output_file

# read an image file
input_stack = stack.Stack(input_filename)

if t_range is None:
    t_slice = slice(0, input_stack.t_count, 1)
else:
    t_slice = slice(t_range[0], t_range[1] + 1, 1)

if channel is None:
    c_slice = [i for i in range(input_stack.c_count)]
else:
    if channel >= 0:
        c_slice = [i for i in range(input_stack.c_count)]
    else:
        c_slice = [i for i in range(input_stack.c_count) if i != channel]

if reverse_channel:
    c_slice = c_slice[::-1]

if z_range is None:
    z_slice = slice(0, input_stack.z_count, 1)
else:
    z_slice = slice(z_range[0], z_range[1] + 1, 1)

if crop_area is None:
    x_slice = slice(0, input_stack.width, 1)
    y_slice = slice(0, input_stack.height, 1)
else:
    x_slice = slice(crop_area[0], crop_area[0] + crop_area[2], 1)
    y_slice = slice(crop_area[1], crop_area[1] + crop_area[3], 1)

# crop TIFF
logging.info("Cropping using area: {0}".format(crop_area))
input_stack.crop([t_slice, c_slice, z_slice, y_slice, x_slice])

# output TIFF
logging.info("Output image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename)
