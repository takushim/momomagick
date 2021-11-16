#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
output_suffix = "_crop.tif"
channel = None
reverse_channel = False
crop_area = None
z_range = None
t_range = None

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

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
channel = args.channel
reverse_channel = args.reverse_channel
z_range = args.z_range
t_range = args.t_range
crop_area = args.crop_area

if args.output_file is None:
    output_filename.append(mmtiff.with_suffix(input_filename, output_suffix))
else:
    output_filename = args.output_file

# read TIFF file (assumes TZ(C)YX order)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

if z_range is None:
    z_slice = slice(0, input_tiff.total_zstack, 1)
else:
    z_slice = slice(z_range[0], z_range[1] + 1, 1)

if t_range is None:
    t_slice = slice(0, input_tiff.total_time, 1)
else:
    t_slice = slice(t_range[0], t_range[1] + 1, 1)

c_slice = np.arange(0, input_tiff.total_channel, 1)
if reverse_channel:
    c_slice = c_slice[::-1]

if channel is not None:
    if channel >= 0:
        c_slice = [c_slice[channel]]
    else:
        indexes = np.arange(0, input_tiff.total_channel, 1)
        c_slice = c_slice[indexes != abs(channel)]

if crop_area is None:
    crop_area = [0, 0, input_tiff.width, input_tiff.height]

# output TIFF
print("Cropping using area:", crop_area)
x_slice, y_slice = mmtiff.area_to_slice(crop_area)
print("Output image:", output_filename)
output_image = input_image[t_slice, z_slice, c_slice, y_slice, x_slice]
input_tiff.save_image(output_filename, output_image)
print(".")
