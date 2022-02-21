#!/usr/bin/env python

import argparse
import numpy as np
from progressbar import progressbar
from mmtools import stack, log

# default values
input_filename = None
output_filename = None
output_suffix = "_proj.tif"
window = 1
axis = 't'
proj_mode = 'average'
proj_list = ['average', 'maximum', 'minimum', 'summation']

# parse arguments
parser = argparse.ArgumentParser(description = 'Project images along the t axis.', \
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help = 'output image filename. [basename]{0} by default'.format(output_suffix))

parser.add_argument('-a', '--axis', default = axis, \
                    help = 'The axis of projection. T, Z or TZ.')

parser.add_argument('-m', '--proj-mode', type = str, default = proj_mode, choices = proj_list, \
                    help = 'Method used to project images.')

parser.add_argument('-w', '--window', type = int, default = window, \
                    help = 'Window of projection.')

parser.add_argument('-c', '--centering', action = 'store_true', \
                    help = 'Project images by setting the current position as the center.')

parser.add_argument('-s', '--sliding', action = 'store_true', \
                    help = 'Project images by sliding the window.')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
proj_mode = args.proj_mode
window = args.window
axis = args.axis.lower()
centering = args.centering
sliding = args.sliding

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# read an image file
input_stack = stack.Stack(input_filename)

# define the function of projection
logger.info("Converting the image to float.")
image_array = input_stack.image_array.astype(float)
if proj_mode == 'average':
    proj_func = np.mean
elif proj_mode == 'maximum':
    proj_func = np.amax
elif proj_mode == 'minimum':
    proj_func = np.amin
elif proj_mode == 'summation':
    proj_func = np.sum

logger.info("Projection function: {0}.".format(proj_func))

if centering:
    def index_to_slice (index):
        return slice(max(0, index - window // 2), index - window // 2 + window, 1)
else:
    def index_to_slice (index):
        return slice(index, index + window, 1)

if sliding:
    def count_to_range (count):
        return range(count)
else:
    def count_to_range (count):
        return range(0, count, window)

if 't' in axis:
    logger.info("T-axis projection. Window: {0}. Sliding: {1}. Centering: {2}.".format(window, sliding, centering))
    image_list = []
    for t_index in progressbar(count_to_range(input_stack.t_count)):
        channel_list = []
        for c_index in range(input_stack.c_count):
            image = proj_func(image_array[index_to_slice(t_index), c_index], axis = 0)
            channel_list.append(image)
        image_list.append(channel_list)
    input_stack.update_array(np.array(np.array(image_list)))

if 'z' in axis:
    logger.info("Z-axis projection. Window: {0}. Sliding: {1}. Centering: {2}.".format(window, sliding, centering))
    image_list = []
    for t_index in progressbar(range(input_stack.t_count)):
        channel_list = []
        for c_index in range(input_stack.c_count):
            image = [proj_func(image_array[t_index, c_index, index_to_slice(z_index)], axis = 0) \
                     for z_index in count_to_range(input_stack.z_count)]
            channel_list.append(image)
        image_list.append(channel_list)
    input_stack.update_array(np.array(np.array(image_list)))

# output TIFF
logger.info("Output image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename, dtype = np.float32)
