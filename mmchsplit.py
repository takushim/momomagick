#!/usr/bin/env python

import os, sys, argparse, re, pathlib, numpy
from mmtools import mmtiff

# default values
input_filename = None
output_folder = None
folder_path_a = "PathA"
folder_path_b = "PathB"

# parse arguments
parser = argparse.ArgumentParser(description='Split PathA and PathB of an image stack', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-folder', default=output_folder, \
                    help='output folder (current folder by default)')

parser.add_argument('input_file', default=input_filename, \
                    help='an input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
if args.output_folder is None:
    output_folder = "."
else:
    output_folder = args.output_folder

# read TIFF file (assumes TZCYX order)
if pathlib.Path(input_filename).exists() is False:
    input_filename = input_filename + ".ome.tif"
    print("Trying to complement the file name as:", input_filename)

input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()
input_prefix = mmtiff.MMTiff.prefix(input_filename)

# set output filenames
output_folder_path_a = pathlib.Path(output_folder).joinpath(folder_path_a)
output_folder_path_b = pathlib.Path(output_folder).joinpath(folder_path_b)
output_filename_path_a = output_folder_path_a.joinpath(input_prefix + "_PathA.ome.tif")
output_filename_path_b = output_folder_path_b.joinpath(input_prefix + "_PathB.ome.tif")

output_folder_path_a.mkdir(exist_ok = True)
output_folder_path_b.mkdir(exist_ok = True)

# output ImageJ, dimensions should be in TZCYX(S) order
image_slice_path_a = input_image[:, :, 0:1]
image_slice_path_b = input_image[:, :, 1:2]
input_tiff.save_image_ome(str(output_filename_path_a), image_slice_path_a)
input_tiff.save_image_ome(str(output_filename_path_b), image_slice_path_b)