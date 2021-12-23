#!/usr/bin/env python

import sys, argparse, itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.interpolation import shift
from mmtools import mmtiff, gpuimage

# default values
input_filenames = None
output_filename = None
filename_suffix = '_sweep.tif'
gpu_id = None
channels = None
z_indexes = None
t_frames = None
shift_range_x = [-20, 20, 0.5]
shift_range_y = [-10, 10, 0.5]

# parse arguments
parser = argparse.ArgumentParser(description='Try overlay of two images using various alignments', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='filename of output TIFF file ([basename]%s by default)' % (filename_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-t', '--t-frames', type = int, nargs = 2, default = t_frames, \
                    help='frames used for overlay (the first frames by default)')

parser.add_argument('-z', '--z-indexes', type = int, nargs = 2, default = z_indexes, \
                    help='z-indexes used for overlay (center by default)')

parser.add_argument('-c', '--channels', type = int, nargs = 2, default = channels, \
                    help='channels used for overlay (the first channels by default)')

group = parser.add_argument_group()
group.add_argument('-x', '--shift-x', type = float, default = None, \
                    help='specify x shift (accepting floats)')

group.add_argument('-X', '--shift-range-x', nargs = 3, type = float, default = shift_range_x, \
                   metavar=('BEGIN', 'END', 'STEP'), \
                   help='range of x shift (accepting floats)')

group = parser.add_argument_group()
group.add_argument('-y', '--shift-y', type = float, default = None, \
                    help='specify y shift (accepting floats)')

group.add_argument('-Y', '--shift-range-y', nargs = 3, type = float, default = shift_range_y, \
                   metavar=('BEGIN', 'END', 'STEP'), \
                   help='range of y shift (accepting floats)')

parser.add_argument('input_files', nargs=2, default=input_filenames, \
                    help='TIFF image files. The first image is overlayed.')
args = parser.parse_args()

# set arguments
gpu_id = args.gpu_id
input_filenames = args.input_files
t_frames = args.t_frames
channels = args.channels
z_indexes = args.z_indexes

if args.shift_x is None:
    shift_range_x = args.shift_range_x
else:
    shift_range_x = [args.shift_x, args.shift_x + 1, 1]

if args.shift_y is None:
    shift_range_y = args.shift_range_y
else:
    shift_range_y = [args.shift_y, args.shift_y + 1, 1]

if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filenames[-1], filename_suffix)
else:
    output_filename = args.output_file

# turn on gpu
if gpu_id is not None:
    gpuimage.turn_on_gpu(gpu_id)

# read TIFF file (assumes TZCYX order)
input_tiffs = [mmtiff.MMTiff(file) for file in input_filenames]

# set values using the image properties
if t_frames is None:
    t_frames = [0, 0]
if channels is None:
    channels = [0, 0]
if z_indexes is None:
    z_indexes = [int(tiff.total_zstack // 2) for tiff in input_tiffs]

# load images
input_images = []
output_shape = (input_tiffs[-1].height, input_tiffs[-1].width)
for index in range(len(input_tiffs)):
    print("Image {0}: t = {1}, c = {2}, z = {3}".format(index, t_frames[index], channels[index], z_indexes[index]))
    image = input_tiffs[index].as_list(list_channel = True)
    image = image[t_frames[index]][channels[index]][z_indexes[index]]
    if image.shape != output_shape:
        image = gpuimage.resize(image, shape = output_shape)
    input_images.append(image.astype(np.float32))

font = ImageFont.truetype(mmtiff.font_path(), output_shape[0] // 16)
font_color = np.max(input_images[-1])

print("X range", shift_range_x)
print("Y range", shift_range_y)
shift_xs = np.arange(shift_range_x[0], shift_range_x[1] + shift_range_x[2], shift_range_x[2])
shift_ys = np.arange(shift_range_y[0], shift_range_y[1] + shift_range_y[2], shift_range_y[2])
output_image_list = []
for (shift_y, shift_x) in itertools.product(shift_ys, shift_xs):
    # prepare output image space
    image_list = []

    # overlay
    shifted_image = gpuimage.shift(input_images[0], (shift_y, shift_x), gpu_id = gpu_id)
    image_list.append(shifted_image)

    # background
    image = Image.fromarray(input_images[-1].copy())
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "X %+04.1f Y %+04.1f" % (shift_x, shift_y), font = font, fill = font_color)
    orig_image = np.array(image)
    image_list.append(orig_image)

    # append
    output_image_list.append(image_list)

# output ImageJ, dimensions should be in TZCYXS order
print("Output image:", output_filename)
output_image = np.array(output_image_list)[np.newaxis]
mmtiff.save_image(output_filename, output_image, \
                  xy_res = 1 / input_tiffs[-1].pixelsize_um, \
                  z_step_um = input_tiffs[-1].z_step_um, \
                  finterval_sec = input_tiffs[-1].finterval_sec)