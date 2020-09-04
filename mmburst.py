#!/usr/bin/env python

import sys, argparse, pathlib, numpy, tifffile
from mmtools import mmtiff

# default values
input_filename = None
output_filename = None
use_channel = 0
burst_frames = 4
filename_suffix = '_burst.tif'

# parse arguments
parser = argparse.ArgumentParser(description='Make burst diSPIM image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output multipage TIFF file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-c', '--use-channel', nargs=1, type=int, default=[use_channel], \
                    help='channel to output')
parser.add_argument('-b', '--burst-frames', nargs=1, type=int, default=[burst_frames], \
                    help='numbers of frames to burst')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
use_channel = args.use_channel[0]
burst_frames = args.burst_frames[0]
if args.output_file is None:
    output_filename = pathlib.Path(input_filename).stem.split('.')[0] + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF file (assumes TZ(C)YX order)
image_file = mmtiff.MMTiff(input_filename)

# remove unnecessary channel(s)
orig_image = image_file.as_array(channel = use_channel, drop_channel = True)
orig_image = orig_image[:, 0, :, :]
total_time, height, width = orig_image.shape

# make burst image
output_image = numpy.zeros((total_time - burst_frames, height, width), dtype = numpy.uint16)
for index in range(total_time - burst_frames + 1):
    output_image[index] = numpy.sum(orig_image[index:(index + burst_frames)], axis = 0)
#output_image = output_image.astype(numpy.uint16)
output_image = output_image[:, numpy.newaxis, numpy.newaxis, :, :]

# output ImageJ, dimensions should be in TZCYXS order
xy_resolution = image_file.pixelsize_um
z_spacing = image_file.z_step_um
print('Output image was shaped into: ',output_image.shape)
tifffile.imsave(output_filename, output_image, imagej = True, \
                resolution = (1 / xy_resolution, 1 / xy_resolution), \
                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

