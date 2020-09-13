#!/usr/bin/env python

import os, sys, argparse, re, pathlib, numpy, tifffile
from mmtools import mmtiff

# default values
input_filename = None
output_prefix = None
use_channel = None

# parse arguments
parser = argparse.ArgumentParser(description='Split an image stack into left and right images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-prefix', nargs=1, default=output_prefix, \
                    help='prefix of output TIFF file ([basename] by default)')

parser.add_argument('-c', '--use-channel', nargs=1, type=int, default=use_channel, \
                    help='select one channel to be output')

parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='an input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
if args.use_channel is not None:
    use_channel = args.use_channel[0]
if args.output_prefix is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('MMStack_Pos[0-9]+$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('_$', '', stem, flags=re.IGNORECASE)
    output_prefix = stem
else:
    output_prefix = args.output_prefix[0]

# read TIFF file (assumes TZCYX order)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()
if use_channel is not None:
    input_image = input_image[:, :, use_channel:(use_channel + 1)]

# split image
split_width = int(input_tiff.width // 2)
if input_tiff.width > split_width * 2:
    print("Input image cannot be split evenly. Input iamge shape:", input_image.shape)
split_images = []
split_images.append(input_image[:, :, :, :, 0:split_width].copy())
split_images.append(input_image[:, :, :, :, split_width:input_tiff.width].copy())

# output ImageJ, dimensions should be in TZCYXS order
for index in range(len(split_images)):
    output_filename = "%s_%d.tif" % (output_prefix, index)
    print("Output image %d:" % (index), split_images[index].shape)
    input_tiff.save_image(output_filename, split_images[index])
