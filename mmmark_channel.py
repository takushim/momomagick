#!/usr/bin/env python

import argparse
import numpy as np
from PIL import Image, ImageDraw
from mmtools import stack, log, particles, gpuimage, draw

# default values
input_filename = None
output_filename = None
output_suffix = '_chmarked.tif'
record_filename = None
record_suffix = '_track.json'
marker_radius = 3
marker_width = 1
marker_color = 'grey'
marker_shape = 'circle'
marker_shape_list = ['circle', 'rectangle', 'cross', 'plus', 'dot']
image_scaling = 1.0
spot_scaling = None
life_lead_offset = 0
life_font_size = 10

# parse arguments
parser = argparse.ArgumentParser(description='Mark detected spots on background images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file ([basename]{0} by default)'.format(output_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-f', '--record-file', default = record_filename, \
                    help='TSV file or TrackJ CSV file ([basename].txt if not specified)')

parser.add_argument('-r', '--marker-radius', type = int, default = marker_radius, \
                    help='radii of markers')

parser.add_argument('-w', '--marker-width', type = int, default = marker_width, \
                    help='line widths of markers')

parser.add_argument('-c', '--marker-color', type = str, default = marker_color, \
                    help='color of markers.')

parser.add_argument('-a', '--marker-shape', type = str, default = marker_shape, choices = marker_shape_list, \
                    help='color of markers.')

parser.add_argument('-x', '--image-scaling', type = float, default = image_scaling, \
                    help='scaling factor of images.')

parser.add_argument('-s', '--spot-scaling', type = float, default = spot_scaling, \
                    help='scaling factor of spot coordinates. scaling factor of images used for None.')

parser.add_argument('-t', '--ignore-time', action = 'store_true', \
                    help='ignore the time parameter and mark spots in all time frames')

parser.add_argument('-z', '--ignore-z-index', action = 'store_true', \
                    help='ignore the z index (useful for marking on a maximum projection)')

parser.add_argument('-l', '--draw-life', action = 'store_true', \
                    help='draw lifetimes for the first spots')

parser.add_argument('-d', '--life-lead-offset', type = int, default = life_lead_offset, \
                    help='offset to draw lifetimes')

parser.add_argument('-n', '--life-font-size', type = int, default = life_font_size, \
                    help='font size to draw lifetimes')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input image file.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# turn on gpu
gpu_id = gpuimage.parse_gpu_argument(args)

# set arguments
input_filename = args.input_file
marker_radius = args.marker_radius
marker_width = args.marker_width
marker_shape = args.marker_shape
marker_color = args.marker_color
image_scaling = args.image_scaling
draw_life = args.draw_life
life_font_size = args.life_font_size
life_lead_offset = args.life_lead_offset
spot_scaling = args.spot_scaling if args.spot_scaling is not None else args.image_scaling
ignore_time = args.ignore_time
ignore_z_index = args.ignore_z_index

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

if args.record_file is None:
    record_filename = stack.with_suffix(input_filename, record_suffix)
else:
    record_filename = args.record_file

# load image and convert into an 8-bit RGB image
logger.info("Loading image: {0}.".format(input_filename))
input_stack = stack.Stack(input_filename)

if np.isclose(image_scaling, 1.0) == False:
    logger.info("Scaling the image by: {0}.".format(image_scaling))
    input_stack.scale_by_ratio(ratio = (1.0, image_scaling, image_scaling), gpu_id = gpu_id, progress = True)

# read the JSON file
logger.info("Reading records: {0}.".format(record_filename))
spot_list = particles.parse_tree(particles.load_spots(record_filename))

# add lifetime as text
for spot in spot_list:
    if spot['parent'] is None:
        count = len([s for s in spot_list if s['track'] == spot['track']])
        spot['text'] = str(count)

if np.isclose(spot_scaling, 1.0) == False:
    logger.info("Scaling spots by: {0}.".format(spot_scaling))
    for spot in spot_list:
        spot['x'] = spot['x'] * spot_scaling
        spot['y'] = spot['y'] * spot_scaling

# marking functions
if ignore_time and ignore_z_index:
    def current_spots (spot_list, t_index, z_index):
        return spot_list
elif ignore_time:
    def current_spots (spot_list, t_index, z_index):
        return [spot for spot in spot_list if spot['z'] == z_index]
elif ignore_z_index:
    def current_spots (spot_list, t_index, z_index):
        return [spot for spot in spot_list if spot['time'] == t_index]
else:
    def current_spots (spot_list, t_index, z_index):
        return [spot for spot in spot_list if spot['time'] == t_index and spot['z'] == z_index]

mark_spots = draw.mark_spots_func(marker_radius, marker_width, marker_shape)
if life_lead_offset > 0:
    draw_texts = draw.draw_texts_func(marker_radius, life_font_size, with_lead = True, \
                                      lead_offset = life_lead_offset, lead_width = marker_width)
else:
    draw_texts = draw.draw_texts_func(marker_radius, life_font_size)

def mark_func (t_index, image_shape, image_dtype):
    image = np.zeros(image_shape, dtype = image_dtype)
    for z_index in range(len(image)):
        z_image = Image.fromarray(image[z_index])
        draw = ImageDraw.Draw(z_image)
        mark_spots(draw, current_spots(spot_list, t_index, z_index), marker_color)
        if draw_life:
            draw_texts(draw, current_spots(spot_list, t_index, z_index), marker_color)
        image[z_index] = np.array(z_image)
    return image

# draw markers
logger.info("Start marking spots.".format(output_filename))
input_stack.extend_channel(mark_func, progress = True, alloc_c_count = 0)

# save image
logger.info("Saving image: {0}.".format(output_filename))
input_stack.save_imagej_tiff(output_filename)