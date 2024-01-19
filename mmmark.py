#!/usr/bin/env python

import argparse
import numpy as np
from PIL import Image, ImageDraw
from mmtools import stack, log, particles, gpuimage, draw

# default values
input_filename = None
output_filename = None
output_suffix = '_marked.tif'
record_filename = None
record_suffix = '_track.json'
marker_radius = 3
marker_width = 1
marker_colors = ['red', 'orange', 'blue']
marker_shape = 'circle'
marker_shape_list = ['circle', 'rectangle', 'cross', 'plus', 'dot']
image_scaling = 1.0
life_lead_offset = 0
life_font_size = 10
spot_scaling = None
spot_shift = [0.0, 0.0, 0.0]

# constants
marker_color = 'gray'
clip_percentile = 0.0

# parse arguments
parser = argparse.ArgumentParser(description='Mark detected spots on background images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file ([basename]{0} by default)'.format(output_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-f', '--record-file', default = record_filename, \
                    help='JSON file recording spots ([basename]{0} by default)'.format(record_suffix))

parser.add_argument('-r', '--marker-radius', type = int, default = marker_radius, \
                    help='radius of markers')

parser.add_argument('-w', '--marker-width', type = int, default = marker_width, \
                    help='line widths of markers')

parser.add_argument('-m', '--marker-colors', nargs = 3, type = str, default = marker_colors, metavar=('NEW', 'CONT', 'END'), \
                    help='marker colors for new, tracked, disappearing spots')

parser.add_argument('-a', '--marker-shape', type = str, default = marker_shape, choices = marker_shape_list, \
                    help='color of markers.')

parser.add_argument('-x', '--image-scaling', type = float, default = image_scaling, \
                    help='scaling factor of images.')

parser.add_argument('-s', '--spot-scaling', type = float, default = spot_scaling, \
                    help='scaling factor of spot coordinates. scaling factor of images used for None.')

parser.add_argument('-S', '--spot-shift', nargs = 3, type = float, default = spot_shift, \
                    metavar = ('X', 'Y', 'Z'), help='Shift the spots.')

parser.add_argument('-l', '--draw-life', action = 'store_true', \
                    help='draw lifetimes for the first spots')

parser.add_argument('-d', '--life-lead-offset', type = int, default = life_lead_offset, \
                    help='offset to draw lifetimes')

parser.add_argument('-n', '--life-font-size', type = int, default = life_font_size, \
                    help='font size to draw lifetimes')

parser.add_argument('-c', '--new-channel', action = 'store_true', \
                    help='draw markers in a new channel. ignores: marker-colors, clip-percentile and invert-lut')

parser.add_argument('-t', '--ignore-time', action = 'store_true', \
                    help='ignore the time parameter and mark spots in all time frames')

parser.add_argument('-z', '--ignore-z-index', action = 'store_true', \
                    help='ignore the z index (useful for marking on a maximum projection)')

parser.add_argument('-i', '--invert-lut', action = 'store_true', \
                    help='invert the LUT of output image')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, help='input image file.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# turn on gpu
gpu_id = gpuimage.parse_gpu_argument(args)

# set arguments
input_filename = args.input_file
marker_radius = args.marker_radius
marker_width = args.marker_width
marker_colors = args.marker_colors
marker_shape = args.marker_shape
invert_lut = args.invert_lut
draw_life = args.draw_life
life_font_size = args.life_font_size
life_lead_offset = args.life_lead_offset
image_scaling = args.image_scaling
new_channel = args.new_channel
spot_scaling = args.spot_scaling if args.spot_scaling is not None else args.image_scaling
spot_shift = args.spot_shift[::-1]
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

# load image and scale them
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
    count = len([s for s in spot_list if s['track'] == spot['track']])
    spot['text'] = str(count)

# shift spots
for pos, shift in zip(['z', 'y', 'x'], spot_shift):
    if np.isclose(shift, 0.0) == False:
        logger.info("Shifting {0} by {1}.".format(pos, shift))
        for spot in spot_list:
            spot[pos] = spot[pos] + shift

# scale spot coordinates if necessary
if np.isclose(spot_scaling, 1.0) == False:
    logger.info("Scaling spots by: {0}.".format(spot_scaling))
    for spot in spot_list:
        spot['x'] = spot['x'] * spot_scaling
        spot['y'] = spot['y'] * spot_scaling

# functions to select spots to draw markers
if ignore_time and ignore_z_index:
    if new_channel:
        def current_spots (spot_list, t_index, z_index):
            return spot_list
    else:
        def current_spots (spot_list, t_index, c_index, z_index):
            return [spot for spot in spot_list if spot['channel'] == c_index]
elif ignore_time:
    if new_channel:
        def current_spots (spot_list, t_index, z_index):
            return [spot for spot in spot_list if spot['z'] == z_index]
    else:
        def current_spots (spot_list, t_index, c_index, z_index):
            return [spot for spot in spot_list if spot['z'] == z_index and spot['channel'] == c_index]
elif ignore_z_index:
    if new_channel:
        def current_spots (spot_list, t_index, z_index):
            return [spot for spot in spot_list if spot['time'] == t_index]
    else:
        def current_spots (spot_list, t_index, c_index, z_index):
            return [spot for spot in spot_list if spot['time'] == t_index and spot['channel'] == c_index]
else:
    if new_channel:
        def current_spots (spot_list, t_index, z_index):
            return [spot for spot in spot_list if spot['time'] == t_index and spot['z'] == z_index]
    else:
        def current_spots (spot_list, t_index, c_index, z_index):
            return [spot for spot in spot_list if spot['time'] == t_index and spot['z'] == z_index and spot['channel'] == c_index]

# drawing functions
mark_spots = draw.mark_spots_func(marker_radius, marker_width, marker_shape)
mark_spots_half = draw.mark_spots_func(marker_radius, marker_width, marker_shape, draw_half = True)
if life_lead_offset > 0:
    draw_texts = draw.draw_texts_func(marker_radius, life_font_size, with_lead = True, \
                                      lead_offset = life_lead_offset, lead_width = marker_width)
else:
    draw_texts = draw.draw_texts_func(marker_radius, life_font_size)

# draw markers
if new_channel:
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

else:
    logger.info("Preparing an RGB uint8 image.")
    input_stack.clip_all(percentile = clip_percentile)
    input_stack.fit_to_uint8(fit_always = False)
    input_stack.add_s_axis(s_count = 3)

    if invert_lut:
        logger.info("Inverting the LUT of the image.")
        input_stack.invert_lut()

    # spot lists
    spots_first = [spot for spot in spot_list if spot['parent'] is None]
    spots_last = [spot for spot in spot_list if (len(particles.find_children(spot, spot_list)) == 0) and (spot not in spots_first)]
    spots_cont = [spot for spot in spot_list if spot not in (spots_first + spots_last)]
    spots_one = [spot for spot in spots_first if (len(particles.find_children(spot, spot_list)) == 0)]

    def mark_func (image, t_index, c_index):
        for z_index in range(len(image)):
            z_image = Image.fromarray(image[z_index])
            draw = ImageDraw.Draw(z_image)

            mark_spots(draw, current_spots(spots_first, t_index, c_index, z_index), marker_colors[0])
            mark_spots(draw, current_spots(spots_cont, t_index, c_index, z_index), marker_colors[1])
            mark_spots(draw, current_spots(spots_last, t_index, c_index, z_index), marker_colors[2])
            mark_spots_half(draw, current_spots(spots_one, t_index, c_index, z_index), marker_colors[2])

            if draw_life:
                draw_texts(draw, current_spots(spots_first, t_index, c_index, z_index), marker_colors[0])

            image[z_index] = np.array(z_image)

        return image

    # draw markers
    logger.info("Marking spots.".format(output_filename))
    input_stack.apply_all(mark_func, progress = True, with_s_axis = True)

# save image
logger.info("Saving image: {0}.".format(output_filename))
input_stack.save_imagej_tiff(output_filename)