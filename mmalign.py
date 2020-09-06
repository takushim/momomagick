#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, tifffile
from scipy.ndimage.interpolation import shift
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import mmtiff

# defaults
input_filename = None
align_filename = 'align.txt'
align_smoothing = False
filename_suffix = '_aligned.tif'
output_filename = None

parser = argparse.ArgumentParser(description='Align a multipage TIFF image according to align.txt', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--align-file', nargs=1, default = [align_filename], \
                    help='aligning tsv file name (align.txt if not specified)')
parser.add_argument('-m', '--align-smoothing', action='store_true', default = align_smoothing, \
                    help='smoothing of alignment curves')

parser.add_argument('-o', '--output-file', nargs=1, default = None, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('input_file', nargs=1, default=None, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
align_filename = args.align_file[0]
align_smoothing = args.align_smoothing

if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file[0]

# read input image(s)
input_tiff = mmtiff.MMTiff(input_filename)
input_image = input_tiff.as_array()

# alignment
move_x = numpy.full(input_tiff.total_time, 0.0)
move_y = numpy.full(input_tiff.total_time, 0.0)

align_table = pandas.read_csv(align_filename, comment = '#', sep = '\t')
print("Using %s for alignment." % (align_filename))
align_plane = numpy.array(align_table.align_plane)
align_x = numpy.array(align_table.align_x)
align_y = numpy.array(align_table.align_y)
if align_smoothing:
    print("Smoothing on.")
    align_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
    align_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)

move_x = move_x - align_x
move_y = move_y - align_y    

# align image
output_image = numpy.zeros_like(input_image)
for time in range(input_tiff.total_time):
    for zstack in range(input_tiff.total_zstack):
        for channel in range(input_tiff.total_channel):
            output_image[time, zstack, channel] = shift(input_image[time, zstack, channel], (move_y[time], move_x[time]))
            print(time, zstack, channel, (align_table.align_x[time], align_table.align_y[time]), (move_x[time], move_y[time]))

# output multipage tiff
print("Output image file to %s." % (output_filename))
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / input_tiff.pixelsize_um, 1 / input_tiff.pixelsize_um), \
                metadata = {'spacing': input_tiff.z_step_um, 'unit': 'um', 'Composite mode': 'composite'})