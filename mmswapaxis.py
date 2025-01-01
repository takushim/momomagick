#!/usr/bin/env python

import argparse
from mmtools import stack, log

# default values
input_filename = None
output_filename = None
output_suffix = "_swap.tif"
swap_axes = [0, 2]

# parse arguments
parser = argparse.ArgumentParser(description='Swap two axes of stack images.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix))

parser.add_argument('-s', '--swap-axes', type = int, default = swap_axes, metavar = "AXIS1, AXIS2", \
                    help='axes to be swapped. default = z t')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input (multipage) TIFF file')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
swap_axes = args.swap_axes

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# read an image file
input_stack = stack.Stack(input_filename)
logger.info("Original axes: {0}".format(input_stack.image_array.shape))

# swap axes
logger.info("Swap: {0}".format(swap_axes))
input_stack.update_array(input_stack.image_array.swapaxes(*swap_axes))
logger.info("Updated axes: {0}".format(input_stack.image_array.shape))

# output TIFF
logger.info("Output image: {0}".format(output_filename))
input_stack.save_ome_tiff(output_filename)
