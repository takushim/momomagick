#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
output_suffix = "_crop.tif"
use_channel = 0
crop_area = None
preset_area_index = 0
preset_areas = mmtiff.preset_areas

# parse arguments
parser = argparse.ArgumentParser(description='Crop a diSPIM image for viewing via network', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix))
parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

group = parser.add_mutually_exclusive_group()
group.add_argument('-P', '--preset-area-index', type = int, default = preset_area_index, \
                   help='Crop using the preset area. ' + \
                        ' '.join(["Area {0}: X {1} Y {2} W {3} H {4}.".format(i, *preset_areas[i]) \
                                  for i in range(len(preset_areas))]))
group.add_argument('-R', '--crop-area', type = int, nargs = 4, default = crop_area, \
                   metavar = ('X', 'Y', 'W', "H"),
                   help='Crop using the specified area.')

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
use_channel = args.use_channel

if args.crop_area is not None:
    crop_area = args.crop_area
elif args.preset_area_index is not None:
    crop_area = preset_areas[args.preset_area_index]

if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)

# read TIFF file (assumes TZ(C)YX order)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

# output TIFF
slice_x, slice_y = mmtiff.area_to_slice(crop_area)
print("Output image:", output_filename)
if use_channel is None:
    output_image = input_image[..., slice_y, slice_x]
else:
    output_image = input_image[..., use_channel:(use_channel + 1), slice_y, slice_x]

input_tiff.save_image(output_filename, output_image)