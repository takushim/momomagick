#!/usr/bin/env python

import sys, argparse
import numpy as np
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
output_suffix = "_crop.tif"
crop_origin = [450, 0]
crop_size = [256, 256]

# parse arguments
parser = argparse.ArgumentParser(description='Crop a diSPIM image for viewing via network', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix))
parser.add_argument('-g', '--crop-origin', nargs=2, type=int, default = crop_origin, \
                    metavar=('X', 'Y'), \
                    help='origin of cropping')
parser.add_argument('-z', '--crop-size', nargs=2, type=int, default = crop_size, \
                    metavar=('WIDTH', 'HEIGHT'), \
                    help='size of cropped image')
parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
crop_origin = args.crop_origin
crop_size = args.crop_size
if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)

# read TIFF file (assumes TZ(C)YX order)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

# output TIFF
print("Output image:", output_filename)
output_image = input_image[..., crop_origin[1]:(crop_origin[1] + crop_size[1]), crop_origin[0]:(crop_origin[0] + crop_size[0])]
input_tiff.save_image(output_filename, output_image)
