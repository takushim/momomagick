#!/usr/bin/env python

import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from progressbar import progressbar
from mmtools import stack, log, particles

# default values
input_filename = None
output_filename = None
output_suffix = '_marked.tif'
record_filename = None
record_suffix = '_track.json'
marker_radius = 3
marker_width = 1
marker_colors = ['red', 'orange', 'blue']

# parse arguments
parser = argparse.ArgumentParser(description='Mark detected spots on background images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file ([basename]{0} by default)'.format(output_suffix))

parser.add_argument('-f', '--record-file', default = record_filename, \
                    help='TSV file or TrackJ CSV file ([basename].txt if not specified)')

parser.add_argument('-r', '--marker-radius', type = int, default = marker_radius, \
                    help='radii of markers')

parser.add_argument('-w', '--marker-width', type = int, default = marker_width, \
                    help='line widths of markers')

parser.add_argument('-m', '--marker-colors', nargs = 3, type=str, default = marker_colors, metavar=('NEW', 'CONT', 'END'), \
                    help='marker colors for new, tracked, disappearing, and redundant spots')

parser.add_argument('-i', '--invert-lut', action = 'store_true', \
                    help='invert the LUT of output image')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input image file.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
marker_radius = args.marker_radius
marker_width = args.marker_width
marker_colors = args.marker_colors
invert_lut = args.invert_lut

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

if args.record_file is None:
    record_filename = stack.with_suffix(input_filename, record_suffix)
else:
    record_filename = args.record_file

# read TIFF files TZCYX(S)
logger.info("Loading image: {0}.".format(input_filename))
input_stack = stack.Stack(input_filename)

# read the JSON file
logger.info("Reading records: {0}.".format(record_filename))
with open(record_filename, 'r') as f:
    json_data = json.load(f)
    spot_list = json_data.get('spot_list', [])
    spot_list = [spot for spot in spot_list if spot['delete'] == False]

# convert image into an 8-bit RGB image
image_array = input_stack.image_array
if input_stack.has_s_axis == False:
    image_array = np.moveaxis(np.array([image_array, image_array, image_array]), 0, -1)
    logger.info("Image array was shaped into: {0}.".format(image_array.shape))

if image_array.dtype != np.uint8:
    logger.info("Converting the pixel values to uint8.")
    image_array = (255.0 * (image_array - np.min(image_array)) / np.ptp(image_array)).astype(np.uint8)

if invert_lut:
    logger.info("Inverting the LUT.")
    image_array = 255 - image_array

# draw spots
def mark_spots (draw, spots, color):
    for spot in spots:
        draw.ellipse((spot['x'] - marker_radius, spot['y'] - marker_radius, spot['x'] + marker_radius, spot['y'] + marker_radius),
                     outline = color, fill = None, width = marker_width)

def mark_ones (draw, spots, color):
    for spot in spots:
        draw.arc((spot['x'] - marker_radius, spot['y'] - marker_radius, spot['x'] + marker_radius, spot['y'] + marker_radius),
                 start = 315, end = 135, fill = color, width = marker_width)

for t_index in progressbar(range(input_stack.t_count)):
    for c_index in range(input_stack.c_count):
        for z_index in range(input_stack.z_count):
            image = Image.fromarray(image_array[t_index, c_index, z_index])
            draw = ImageDraw.Draw(image)

            spots_current = [spot for spot in spot_list \
                             if spot['time'] == t_index and spot['channel'] == c_index and spot['z'] == z_index]

            spots_first = [spot for spot in spots_current if spot['parent'] is None]
            spots_last = [spot for spot in spots_current
                          if (len(particles.find_children(spot, spot_list)) == 0) and (spot not in spots_first)]
            spots_cont = [spot for spot in spots_current if spot not in (spots_first + spots_last)]
            spots_one = [spot for spot in spots_first if (len(particles.find_children(spot, spot_list)) == 0)]

            mark_spots(draw, spots_first, marker_colors[0])
            mark_spots(draw, spots_cont, marker_colors[1])
            mark_spots(draw, spots_last, marker_colors[2])
            mark_ones(draw, spots_one, marker_colors[2])

            image_array[t_index, c_index, z_index] = np.array(image)

input_stack.update_array(image_array)

# save image
logger.info("Saving image: {0}.".format(output_filename))
input_stack.save_imagej_tiff(output_filename)