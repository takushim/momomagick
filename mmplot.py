#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas
from mmtools import mmtiff, lifetime, spotmark

# prepare spot marker
spotmarker = spotmark.SpotMark()
lifetime_analyzer = lifetime.Lifetime()

# default values
input_filename = None
output_filename = None
marker_filename = 'spot_table.txt'
mask_filename = None
shift_x = 0
shift_y = 0
filename_suffix = '_plotted.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Plot fluorescent puncta to a background image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))

parser.add_argument('-s', '--image-shift', nargs=2, type=int, default=[shift_x, shift_y], \
                    metavar=('X', 'Y'), \
                    help='shift of the image against spots')

parser.add_argument('-f', '--marker-file', default=marker_filename, \
                    help='name of TSV file (read [basename].txt if not specified)')
parser.add_argument('-z', '--marker-size', type=int, default=spotmarker.marker_size, \
                    help='marker size to draw')
parser.add_argument('-c', '--marker-colors', nargs=4, type=str, default=spotmarker.marker_colors, \
                    metavar=('NEW', 'CONT', 'END', 'REDUN'), \
                    help='marker colors for new, tracked, disappearing, and redundant spots')
parser.add_argument('-r', '--rainbow-colors', action='store_true', default=spotmarker.marker_rainbow, \
                    help='use rainbow colors to distinguish each tracking')

parser.add_argument('-m', '--mask-image', default = mask_filename, \
                    help='read masking image to omit unnecessary area')

parser.add_argument('-i', '--invert-image', action='store_true', default=spotmarker.invert_image, \
                    help='invert the LUT of output image')

parser.add_argument('input_file', default=input_filename, \
                    help='input single or multi-page TIFF file to plot spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
mask_filename = args.mask_image
shift_x, shift_y = args.image_shift
marker_filename = args.marker_file
spotmarker.marker_size = args.marker_size
spotmarker.marker_colors = args.marker_colors
spotmarker.marker_rainbow = args.rainbow_colors
spotmarker.invert_image = args.invert_image

if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

# read TSV file
print("Read spots from %s." % (marker_filename))
spot_table = pandas.read_csv(marker_filename, comment = '#', sep = '\t')
total_planes = spot_table.plane.max() + 1

# shift spots
print("Shifting the stamp image:", shift_x, shift_y)
spot_table['x'] = spot_table['x'] - shift_x
spot_table['y'] = spot_table['y'] - shift_y

# read TIFF files TZCYX(S)
#input_image = tifffile.imread(input_filename)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()[:, 0, 0]
height, width = input_tiff.height, input_tiff.width

# filter spots with a masking image
if mask_filename is not None:
    mask_image = tifffile.imread(mask_filename)
    total_spots = len(spot_table)
    spot_table = lifetime_analyzer.filter_spots_maskimage(spot_table, mask_image)
    print("Filtered %d spots using a mask image: %s." % (total_spots - len(spot_table), mask_filename))

# make an output image
output_image = numpy.zeros((total_planes, height, width, 3), dtype = numpy.uint8)
if input_tiff.total_time > 1:
    if input_tiff.colored is True:
        for index in range(len(output_image)):
            output_image[index] = input_image[index]
    else:
        for index in range(len(output_image)):
            output_image[index, :, :, 0] = input_image[index, :, :]
            output_image[index, :, :, 1] = input_image[index, :, :]
            output_image[index, :, :, 2] = input_image[index, :, :]
else:
    for index in range(len(output_image)):
        #output_image[index] = input_image
        output_image[index, :, :, 0] = input_image
        output_image[index, :, :, 1] = input_image
        output_image[index, :, :, 2] = input_image
print("Marked %d spots on %s." % (len(spot_table), input_filename))
image_color = spotmarker.mark_spots(output_image, spot_table)

# output ImageJ, dimensions should be in TZCYXS order
print('Output image was shaped into:', output_image.shape)
input_tiff.save_image(output_filename, output_image)

