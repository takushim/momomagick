#!/usr/bin/env python

import argparse
import numpy as np
from PIL import Image, ImageDraw
from mmtools import stack, log, particles, gpuimage, draw

# default values
input_filename = None
output_filename = None
output_suffix = '_colmarked.tif'
record_filename = None
record_suffix = '_track.json'
marker_radius = 3
marker_width = 1
marker_colors = ['red', 'orange', 'blue']
clip_percentile = 0.0
image_scaling = 1.0
spot_scaling = None

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

parser.add_argument('-m', '--marker-colors', nargs = 3, type = str, default = marker_colors, metavar=('NEW', 'CONT', 'END'), \
                    help='marker colors for new, tracked, disappearing, and redundant spots')

parser.add_argument('-c', '--clip-percentile', type = float, default = clip_percentile, \
                    help='percentile used for automatic clipping. ignored for images within uint8 range.')

parser.add_argument('-x', '--image-scaling', type = float, default = image_scaling, \
                    help='scaling factor of images.')

parser.add_argument('-s', '--spot-scaling', type = float, default = spot_scaling, \
                    help='scaling factor of spot coordinates. scaling factor of images used for None.')

parser.add_argument('-l', '--draw-life', action = 'store_true', \
                    help='draw lifetimes for the first spots')

parser.add_argument('-i', '--invert-lut', action = 'store_true', \
                    help='invert the LUT of output image')

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
marker_colors = args.marker_colors
invert_lut = args.invert_lut
draw_life = args.draw_life
clip_percentile = args.clip_percentile
image_scaling = args.image_scaling
spot_scaling = args.spot_scaling if args.spot_scaling is not None else args.image_scaling

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
logger.info("Preparing an RGB uint8 image.")
input_stack.clip_all(percentile = clip_percentile)
input_stack.fit_to_uint8(fit_always = False)

if invert_lut:
    logger.info("Inverting the LUT of the image.")
    input_stack.invert_lut()

if np.isclose(image_scaling, 1.0) == False:
    logger.info("Scaling the image by: {0}.".format(image_scaling))
    input_stack.scale_by_ratio(ratio = (1.0, image_scaling, image_scaling), gpu_id = gpu_id, progress = True)

input_stack.add_s_axis(s_count = 3)

# read the JSON file
logger.info("Reading records: {0}.".format(record_filename))
spot_list = particles.parse_tree(particles.load_spots(record_filename))

if np.isclose(spot_scaling, 1.0) == False:
    logger.info("Scaling spots by: {0}.".format(spot_scaling))
    for spot in spot_list:
        spot['x'] = spot['x'] * spot_scaling
        spot['y'] = spot['y'] * spot_scaling

# spot lists
spots_first = [spot for spot in spot_list if spot['parent'] is None]
spots_last = [spot for spot in spot_list if (len(particles.find_children(spot, spot_list)) == 0) and (spot not in spots_first)]
spots_cont = [spot for spot in spot_list if spot not in (spots_first + spots_last)]
spots_one = [spot for spot in spots_first if (len(particles.find_children(spot, spot_list)) == 0)]

# add lifetime as text
for spot_first in spots_first:
    count = len([spot for spot in spot_list if spot['track'] == spot_first['track']])
    spot_first['text'] = str(count)

# marking functions
mark_spots = draw.mark_spots_func(marker_radius, marker_width)
mark_ones = draw.mark_spots_func(marker_radius, marker_width, shape = 'arc')
draw_texts = draw.draw_texts_func(marker_radius, 10)

def current_spots (spot_list, t_index, c_index, z_index):
    return [spot for spot in spot_list if spot['time'] == t_index and spot['channel'] == c_index and spot['z'] == z_index]

def mark_func (image, t_index, c_index):
    for z_index in range(len(image)):
        z_image = Image.fromarray(image[z_index])
        draw = ImageDraw.Draw(z_image)

        mark_spots(draw, current_spots(spots_first, t_index, c_index, z_index), marker_colors[0])
        mark_spots(draw, current_spots(spots_cont, t_index, c_index, z_index), marker_colors[1])
        mark_spots(draw, current_spots(spots_last, t_index, c_index, z_index), marker_colors[2])
        mark_ones(draw, current_spots(spots_one, t_index, c_index, z_index), marker_colors[2])

        if draw_life:
            draw_texts(draw, current_spots(spots_first, t_index, c_index, z_index), marker_colors[0])

        image[z_index] = np.array(z_image)

    return image

# draw markers
logger.info("Start marking spots.".format(output_filename))
input_stack.apply_all(mark_func, progress = True, with_s_axis = True)

# save image
logger.info("Saving image: {0}.".format(output_filename))
input_stack.save_imagej_tiff(output_filename)