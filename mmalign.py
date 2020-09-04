#!/usr/bin/env python

import os, platform, sys, glob, argparse
import numpy, pandas, tifffile
from scipy.ndimage.interpolation import shift

# defaults
input_filenames = None
align_spots = True
align_filename = 'align.txt'

output_image_filename = None

parser = argparse.ArgumentParser(description='Align fluorescent-spot image according to align.txt', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n', '--no-align', action='store_true', default=(align_spots is False), \
                    help='plot without alignment')
parser.add_argument('-f', '--align-filename', nargs=1, default = [align_filename], \
                    help='aligning tsv file name (align.txt if not specified)')

parser.add_argument('-o', '--output-image-file', nargs=1, default = None, \
                    help='output image file name([basename]_spotalign.tif if not specified)')

parser.add_argument('input_file', nargs='+', default=None, \
                    help='input multpage-tiff file(s) to align')
args = parser.parse_args()

# collect input filenames
if (platform.system() == "Windows"):
    input_filenames = []
    for pattern in args.input_file:
        input_filenames.extend(sorted(glob.glob(pattern)))
else:
    input_filenames = args.input_file

# set arguments
align_spots = (args.no_align is False)
align_filename = args.align_filename[0]

if args.output_image_file is None:
    output_image_filename = os.path.splitext(os.path.basename(input_filenames[0]))[0] + '_spotalign.tif'
    if output_image_filename in args.input_file:
        raise Exception('input_filename == output_filename')
else:
    output_image_filename = args.output_image_file[0]

# read input image(s)
image_list = []
for input_filename in input_filenames:
    images = tifffile.imread(input_filename)
    if len(images.shape) == 2:
        image_list += [images]
    else:
        image_list += [images[i] for i in range(len(images))]

orig_images = numpy.asarray(image_list)

# alignment
if align_spots is True:
    align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
    print("Using %s for alignment." % (align_filename))
    align_plane = numpy.array(align_table.align_plane)
    align_x = numpy.array(align_table.align_x)
    align_y = numpy.array(align_table.align_y)
else:
    print("Skip using alignment")

# align image
output_image_array = numpy.zeros_like(orig_images)
for index in range(len(orig_images)):
    if align_spots is True:
        align_index = numpy.where(align_plane == index)[0][0]
        shift_x = - round(align_x[align_index])
        shift_y = - round(align_y[align_index])
        print("Plane %d, Shift_X %d, Shift_Y %d" % (align_index, shift_x, shift_y))
    else:
        shift_x = 0
        shift_y = 0
    output_image_array[index] = shift(orig_images[index], [shift_y, shift_x], cval = 0)

# output multipage tiff
print("Output image file to %s." % (output_image_filename))
tifffile.imsave(output_image_filename, output_image_array)
