#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, itertools
from scipy.ndimage.interpolation import shift
from scipy.ndimage import rotate
from mmtools import mmtiff

# defaults
input_filename = None
filename_suffix = '_rotated.tif'
output_filename = None

parser = argparse.ArgumentParser(description='Rotate a multipage TIFF image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read input image(s)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

# prepare an empty array
output_images = []
for angle in [0, 30]:
    print("Rotation:", angle)
    output_images.append(rotate(input_image, angle, axes = (1, 2), reshape = False))
    print(output_images[-1].shape)

output_array = numpy.array(output_images)
#output_array = output_array[:, :, numpy.newaxis, :, :]

# output multipage tiff, dimensions should be in TZCYX order
print("Output image file to %s." % (output_filename))
input_tiff.save_image(output_filename, output_array)
