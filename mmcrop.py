#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
output_suffix = "_crop_{0}.tif"
use_channel = 0
use_area = None
preset_area_index = None
preset_areas = mmtiff.preset_areas
zstack_range = None
time_range = None

# parse arguments
parser = argparse.ArgumentParser(description='Crop a diSPIM image for viewing via network. Crop using all preset areas by default.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix.format('##')))
parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

group = parser.add_mutually_exclusive_group()
group.add_argument('-P', '--preset-area-index', type = int, default = preset_area_index, \
                   help='Crop using the preset area. ' + \
                        ' '.join(["Area {0}: X {1} Y {2} W {3} H {4}.".format(i, *preset_areas[i]) \
                                  for i in range(len(preset_areas))]))
group.add_argument('-R', '--use-area', type = int, nargs = 4, default = use_area, \
                   metavar = ('X', 'Y', 'W', "H"),
                   help='Crop using the specified area.')

parser.add_argument('-Z', '--zstack-range', type = int, nargs = 2, default = zstack_range, \
                    metavar = ('START', 'END'),
                    help='Specify the range of z planes to output')

parser.add_argument('-T', '--time-range', type = int, nargs = 2, default = time_range, \
                    metavar = ('START', 'END'),
                    help='Specify the range of time frames to output')

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
use_channel = args.use_channel
zstack_range = args.zstack_range
time_range = args.time_range

if args.use_area is not None:
    crop_areas = [args.use_area]
elif args.preset_area_index is not None:
    crop_areas = [preset_areas[args.preset_area_index]]
else:
    crop_areas = preset_areas

output_filenames = []
if args.output_file is None:
    for index in range(len(crop_areas)):
        output_filenames.append(mmtiff.with_suffix(input_filename, output_suffix.format(index)))
else:
    if len(crop_areas) > 1:
        print("More than one areas are cropped. Filenames are automatically generated.")
        for index in range(len(crop_areas)):
            output_filenames.append(mmtiff.with_suffix(input_filename, output_suffix.format(index)))
    else:
        output_filenames.append(args.output_file)

# read TIFF file (assumes TZ(C)YX order)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

if zstack_range is None:
    zstack_slice = slice(0, input_tiff.total_zstack, 1)
else:
    zstack_slice = slice(zstack_range[0], zstack_range[1] + 1, 1)

if time_range is None:
    time_slice = slice(0, input_tiff.total_time, 1)
else:
    time_slice = slice(time_range[0], time_range[1] + 1, 1)

if use_channel is None:
    channel_slice = slice(0, input_tiff.total_channel, 1)
else:
    channel_slice = slice(use_channel, use_channel + 1, 1)

# output TIFF
for index in range(len(crop_areas)):
    print("Cropping using area:", crop_areas[index])
    x_slice, y_slice = mmtiff.area_to_slice(crop_areas[index])
    print("Output image:", output_filenames[index])
    output_image = input_image[time_slice, zstack_slice, channel_slice, y_slice, x_slice]

    input_tiff.save_image(output_filenames[index], output_image)
    print(".")
