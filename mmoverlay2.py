#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, tifffile
from scipy.ndimage.interpolation import shift
from mmtools import mmtiff

# default values
input_filenames = None
output_filename = None
shift_x = 0
shift_y = 0
filename_suffix = '_overlay.tif'
xy_resolution = 0.1625
z_spacing = 0.5

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two diSPIM images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='ajustment for overlay')
parser.add_argument('input_file', nargs=2, default=input_filenames, \
                    help='two input (multipage) TIFF files (image_0, image_1)')
args = parser.parse_args()

# set arguments
input_filenames = args.input_file
shift_x, shift_y = args.image_shift
if args.output_file is None:
    stem = pathlib.Path(input_filenames[0]).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filenames[0] == output_filename:
        raise Exception('input_filename == output_filename.')
    if input_filenames[1] == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF files
input_tiffs = []
input_tiffs.append(mmtiff.MMTiff(input_filenames[0]))
input_tiffs.append(mmtiff.MMTiff(input_filenames[1]))
if any([x.colored is True for x in input_tiffs]):
    raise Exception('Color images are not accepted.')

input_images = []
input_images.append(input_tiffs[0].as_array())
input_images.append(input_tiffs[1].as_array())

# shift
input_images[0] = shift(input_images[0], (0, 0, 0, shift_y, shift_x))

# output ImageJ, dimensions should be in TZCYXS order
output_image = numpy.array(input_images)
print('Output image was shaped into:', output_image.shape)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

