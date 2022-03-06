#!/usr/bin/env python

import argparse, platform
import numpy as np
from itertools import product
from progressbar import progressbar
from PIL import Image, ImageDraw, ImageFont
from mmtools import stack, log, gpuimage

# default values
input_filenames = None
output_filename = None
output_suffix = '_sweep.tif'
t_frames = [0, 0]
channels = [0, 0]
z_indexes = None
shift_range_x = [-20, 20, 0.5]
shift_range_y = [-10, 10, 0.5]

# parse arguments
parser = argparse.ArgumentParser(description='Try overlay of two images using various alignments', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='filename of output TIFF file ([basename]%s by default)' % (output_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-t', '--t-frames', type = int, nargs = 2, default = t_frames, \
                    help='frames used for overlay (the first frames by default)')

parser.add_argument('-c', '--channels', type = int, nargs = 2, default = channels, \
                    help='channels used for overlay (the first channels by default)')

parser.add_argument('-z', '--z-indexes', type = int, nargs = 2, default = z_indexes, \
                    help='z-indexes used for overlay (center by default)')

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

log.add_argument(parser)

parser.add_argument('input_files', nargs=2, default=input_filenames, \
                    help='TIFF image files. The first image is overlayed.')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# turn on gpu
gpu_id = gpuimage.parse_gpu_argument(args)

# set arguments
input_filenames = args.input_files
t_frames = args.t_frames
channels = args.channels
shift_range_x = [args.shift_x, args.shift_x + 1, 1] if args.shift_range_x is None else args.shift_range_x
shift_range_y = [args.shift_y, args.shift_y + 1, 1] if args.shift_range_y is None else args.shift_range_y

if args.output_file is None:
    output_filename = stack.with_suffix(input_filenames[1], output_suffix)
else:
    output_filename = args.output_file

if platform.system() == "Windows":
    font_filename = 'C:/Windows/Fonts/Arial.ttf'
elif platform.system() == "Linux":
    font_filename = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
elif platform.system() == "Darwin":
    font_filename = '/Library/Fonts/Verdana.ttf'
else:
    raise Exception('Unknown operating system. Font cannot be loaded.')

# read images
input_stacks = [stack.Stack(file) for file in input_filenames]

# set values using the image properties
if z_indexes is None:
    z_indexes = [int(stack.z_count // 2) for stack in input_stacks]

# allocate output image
shift_xs = np.arange(shift_range_x[0], shift_range_x[1] + shift_range_x[2], shift_range_x[2])
shift_ys = np.arange(shift_range_y[0], shift_range_y[1] + shift_range_y[2], shift_range_y[2])
logger.info("X shift range: {0}".format(shift_range_x))
logger.info("Y shift range: {0}".format(shift_range_y))

output_stack = stack.Stack()
output_shape = list(input_stacks[1].image_array.shape)
output_shape[0] = len(shift_xs) * len(shift_ys)
output_shape[1] = 2
output_shape[2] = 1
output_stack.alloc_zero_image(output_shape, dtype = np.float, \
                              voxel_um = input_stacks[1].voxel_um, \
                              finterval_sec = input_stacks[1].finterval_sec)

font = ImageFont.truetype(font_filename, max(output_stack.height // 16, 16))

for index, (shift_y, shift_x) in progressbar(enumerate(product(shift_ys, shift_xs)), max_value = output_shape[0]):
    # background
    image = input_stacks[0].image_array[t_frames[0], channels[0], z_indexes[0]].astype(float)
    font_color = image.max()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), "X %+04.1f Y %+04.1f" % (shift_x, shift_y), font = font, fill = font_color)
    output_stack.image_array[index, 0] = np.array(image)

    # overlay
    image = input_stacks[1].image_array[t_frames[1], channels[1], z_indexes[1]].astype(float)
    image = gpuimage.shift(image, (shift_y, shift_x), gpu_id = gpu_id)
    output_stack.image_array[index, 1] = image


# output image
logger.info("Saving image: {0}.".format(output_filename))
output_stack.save_ome_tiff(output_filename, dtype = np.float32)
