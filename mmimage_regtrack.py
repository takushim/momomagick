#!/usr/bin/env python

import argparse, json
import numpy as np
from mmtools import gpuimage, stack, log, particles, register

# defaults
input_filename = None
record_filename = None
record_suffix = '_track.json'
output_filename = None
output_suffix = '_mreg.tif'

parser = argparse.ArgumentParser(description='Register time-lapse images using a tracking record', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-f', '--record-file', default = record_filename, \
                    help='filename of json tracking record ([basename]{0} if not specified)'.format(record_suffix))

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# turn on gpu
gpu_id = gpuimage.parse_gpu_argument(args)

# set arguments
input_filename = args.input_file

if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

if args.record_file is None:
    record_filename = stack.with_suffix(input_filename, record_suffix)
else:
    record_filename = args.record_file

# read input image
logger.info("Loading image: {0}.".format(input_filename))
input_stack = stack.Stack(input_filename, keep_s_axis = False)

# read record file
logger.info("Loading record: {0}.".format(record_filename))
spot_list = particles.load_spots(record_filename)
track_list = particles.parse_tree(spot_list)
if max([track['track'] for track in track_list]) > 0:
    Exception('Tracking record should contain only one track.')
track_list = [track for track in track_list if track['track'] == 0]

# prepare a shift list
logger.info("Prepareing a list of image shift.")
shift_list = []
for t_index in range(input_stack.t_count):
    tracks = [track for track in track_list if track['time'] == t_index]
    if len(tracks) == 0:
        logger.warning('No track record found for t = {0}.'.format(t_index))
        shift = [0.0, 0.0, 0.0]
    else:
        if len(tracks) > 1:
            logger.warning("More than one track record found for t = {0}.".format(t_index))
        shift = [tracks[0]['z'], tracks[0]['y'], tracks[0]['x']]
    shift_list.append(shift)

ref_shift = shift_list[min([track['time'] for track in track_list])]

logger.info("Adjusting all shifts using a reference shift: {0}".format(ref_shift))
shift_list = [[shift[0] - ref_shift[0], shift[1] - ref_shift[1], shift[2] - ref_shift[2]] for shift in shift_list]

# align image.
logger.info("Preparing an aligned image: {0}.".format(output_filename))
if input_stack.z_count > 1:
    def affine_transform (image, t_index, c_index):
        matrix = register.drift_to_matrix_3d(shift_list[t_index])
        return gpuimage.affine_transform(image, matrix, gpu_id = gpu_id)
else:
    def affine_transform (image, t_index, c_index):
        matrix = register.drift_to_matrix_2d(shift_list[t_index][1:])
        return gpuimage.affine_transform(image[0], matrix, gpu_id = gpu_id)[np.newaxis]

input_stack.apply_all(affine_transform, progress = True)

# output filename
logger.info("Output an image: {0}.".format(output_filename))
input_stack.save_ome_tiff(output_filename)

