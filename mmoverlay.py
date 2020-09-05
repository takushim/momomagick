#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, tifffile
from scipy.ndimage.interpolation import shift
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import mmtiff

# default values
input_filenames = None
output_filename = None
align_images = False
align_filename = 'align.txt'
align_smoothing = False
shift_x = 0
shift_y = 0
filename_suffix = '_overlay.tif'
xy_resolution = 0.1625
z_spacing = 0.5

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two diSPIM images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='ajustment for overlay')

parser.add_argument('-A', '--align-images', action='store_true', default = align_images, \
                    help='align overlay images using a TSV file')
parser.add_argument('-f', '--align-filename', nargs=1, default = [align_filename], \
                    help='aligning tsv file name ({0} if not specified)'.format(align_filename))
parser.add_argument('-m', '--align-smoothing', action='store_true', default = align_smoothing, \
                    help='smoothing of alignment curves')

parser.add_argument('input_file', nargs=2, default=input_filenames, \
                    help='two input (multipage) TIFF files (overlay, background)')
args = parser.parse_args()

# set arguments
input_filenames = args.input_file
shift_x, shift_y = args.image_shift
align_images = args.align_images
align_filename = args.align_filename[0]
align_smoothing = args.align_smoothing
if args.output_file is None:
    stem = pathlib.Path(input_filenames[0]).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('_[0-9]+$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if any([x == output_filename for x in input_filenames]):
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF files
input_tiffs = []
input_tiffs.append(mmtiff.MMTiff(input_filenames[0]))
input_tiffs.append(mmtiff.MMTiff(input_filenames[1]))
#if any([x.colored is True for x in input_tiffs]):
#    raise Exception('Color images are not accepted.')

input_images = []
input_images.append(input_tiffs[0].as_array())
input_images.append(input_tiffs[1].as_array())

# determine image size and dtype that can store both of the images
overlay_width = max([x.width for x in input_tiffs])
overlay_height = max([x.height for x in input_tiffs])
overlay_zstack = max([x.total_zstack for x in input_tiffs])
overlay_time = max([x.total_time for x in input_tiffs])
if any([x.dtype == numpy.uint32 for x in input_tiffs]):
    output_dtype = numpy.uint32
elif any([x.dtype == numpy.uint16 for x in input_tiffs]):
    output_dtype = numpy.uint16
else:
    output_dtype = numpy.uint8

# read alignment
move_x = numpy.full(input_tiffs[0].total_time, shift_x)
move_y = numpy.full(input_tiffs[0].total_time, shift_y)
if align_images:
    align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
    print("Using {0} for alignment.".format(align_filename))
    align_plane = numpy.array(align_table.align_plane)
    align_x = numpy.array(align_table.align_x)
    align_y = numpy.array(align_table.align_y)
    if align_smoothing:
        print("Smoothing on.")
        align_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
        align_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)
    move_x = move_x - align_x
    move_y = move_y - align_y

# list of output_images (list by channel)
output_images = []

# shift
for index in range(input_tiffs[0].total_channel):
    image = numpy.zeros((overlay_time, overlay_zstack, overlay_height, overlay_width), dtype = output_dtype)
    image[:] = input_images[0][:, :, index].copy()
    for time in range(input_tiffs[0].total_time):
        for zstack in range(input_tiffs[0].total_zstack):
            image[time, zstack] = shift(image[time, zstack], (move_y[time], move_x[time]))
            print("Time, stack, move", time, zstack, (move_y[time], move_x[time]))
    output_images.append(image)
    print("Shifted channel", index, "of Image 0.")

# background
for index in range(input_tiffs[1].total_channel):
    image = numpy.zeros((overlay_time, overlay_zstack, overlay_height, overlay_width), dtype = output_dtype)
    image[:] = input_images[1][:, :, index].copy()
    if input_tiffs[1].total_time == 1:
        print("Broadcasting the background image")
    output_images.append(image)
    print("Concatenated channel", index, "of Image 1.")

# output ImageJ, dimensions should be in TZCYXS order
output_images = numpy.array(output_images)
trans = numpy.arange(output_images.ndim)
trans[0], trans[1], trans[2] = 1, 2, 0
output_images = output_images.transpose(trans)[:, :, ::-1]

print('Output image was shaped into:', output_images.shape)
tifffile.imsave(output_filename, output_images, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

