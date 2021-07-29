#!/usr/bin/env python

import os, sys, argparse, re, pathlib, numpy
from mmtools import mmtiff

# default values
input_filename = None
output_folder = None
channel_prefix = "path"

# parse arguments
parser = argparse.ArgumentParser(description='Split PathA and PathB of an image stack', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-folder', default=output_folder, \
                    help='output folder (current folder by default)')

parser.add_argument('-c', '--channel-prefix', default=channel_prefix, \
                    help='prefix to the channel folders ([perfix]_[number])')

parser.add_argument('input_file', default=input_filename, \
                    help='an input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
channel_prefix = args.channel_prefix
if args.output_folder is None:
    output_folder = "."
else:
    output_folder = args.output_folder

# read TIFF file (assumes TZCYX order)
if pathlib.Path(input_filename).exists() == False:
    input_filename = input_filename + ".ome.tif"
    print("Trying to complement the file name as:", input_filename)

input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()
input_prefix = mmtiff.prefix(input_filename)

# set output filenames
for channel in range(input_tiff.total_channel):
    channel_folder = "{0}_{1}".format(channel_prefix, channel)
    channel_path = pathlib.Path(output_folder).joinpath(channel_folder)
    channel_filename = "{0}_{1}_{2}.ome.tif".format(input_prefix, channel_prefix, channel)
    channel_fullpath = channel_path.joinpath(channel_filename)

    # make an channel folder
    channel_path.mkdir(exist_ok = True)

    # output ImageJ, dimensions should be in TZCYX(S) order
    print("Output:", channel_filename)
    image_slice = input_image[:, :, channel:(channel + 1)]
    input_tiff.save_image_ome(str(channel_fullpath), image_slice)
