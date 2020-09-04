#!/usr/bin/env python

import os, sys, argparse, re, pathlib, numpy, tifffile
from mmtools import mmtiff

# default values
input_filename = None
output_prefix = None

# parse arguments
parser = argparse.ArgumentParser(description='Split an image stack into left and right images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-prefix', nargs=1, default=output_prefix, \
                    help='prefix of output TIFF file ([basename] by default)')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='an input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
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

# split image
split_width = input_tiff.width // 2
split_shape = numpy.array(input_image.shape)
split_shape[-1] = split_width
if input_tiff.width > split_width * 2:
    print("Input image cannot be split evenly. Input iamge shape:", input_image.shape)
split_images = []
split_images.append(input_image[:, :, :, :, 0:split_width].copy())
split_images.append(input_image[:, :, :, :, split_width:input_tiff.width].copy())

# output ImageJ, dimensions should be in TZCYXS order
xy_resolution = input_tiff.pixelsize_um
z_spacing = input_tiff.z_step_um

for index in range(len(split_images)):
    output_filename = "%s_%d.tif" % (output_prefix, index)
    tifffile.imsave(output_filename, split_images[index], imagej = True, \
                    resolution = (1 / xy_resolution, 1 / xy_resolution), \
                    metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})
    print("Output image %d:" % (index), split_images[index].shape)
