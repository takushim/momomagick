#!/usr/bin/env python

import argparse, json, copy
import numpy as np
from datetime import datetime
from pathlib import Path
from numpyencoder import NumpyEncoder
from progressbar import progressbar
from mmtools import stack, gpuimage, register, log

# default values
input_filenames = None
channels = [0, 0]
init_flip = ''
init_rot = [0.0, 0.0, 0.0]
init_shift = [0.0, 0.0, 0.0]
post_shift = [0.0, 0.0, 0.0]
output_image_filename = None
output_image_suffix = '_overlay.tif' # overwritten by the registration method
output_json_filename = None
output_json_suffix = '_overlay.json' # overwritten by the registration method
truncate_frames = False
scale_image = None
register_all = False
reg_method = 'Full'
reg_method_list = register.registering_methods
opt_method = "Powell"
opt_method_list = register.optimizing_methods

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two time-lapse images after registration', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-image-file', default=output_image_filename, \
                    help='output TIFF file ([basename]{0} by default)'.format(output_image_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-J', '--output-json', action = 'store_true', \
                    help='Enable output of the json file')

parser.add_argument('-j', '--output-json-file', default = output_json_filename, \
                    help='output JSON file name ([basename]{0} by default)'.format(output_json_suffix))

parser.add_argument('-c', '--channels', nargs = 2, type = int, default = channels, \
                    help='specify two channels used for registration')

parser.add_argument('-a', '--register-all', action = 'store_true', \
                    help='Perform registration for each pair of images')

parser.add_argument('-n', '--truncate-frames', action = 'store_true', \
                    help='Disable broadcasting and truncate frames')

parser.add_argument('-e', '--reg-method', type = str, default = reg_method, choices = reg_method_list, \
                    help='Method used for registration')

parser.add_argument('-t', '--opt-method', type = str, default = opt_method, choices = opt_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-F', '--init-flip', type = str, default = init_flip, \
                    help='Flip the overlay image (e.g., X, XZ, XYZ)')

parser.add_argument('-R', '--init-rot', nargs = 3, type = float, default = init_rot, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Rotate the overlay image. Applied after flip. Consider isometric scaling.')

parser.add_argument('-S', '--init-shift', nargs = 3, type = float, default = init_shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Shift the overlay image. Applied after rotation.')

parser.add_argument('-P', '--post-shift', nargs = 3, type = float, default = post_shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Post-registration shift of the overlay image (for adjustment")')

parser.add_argument('-s', '--scale-isometric', action = 'store_true', \
                    help='Scale images to achieve isotrophic voxels')

parser.add_argument('-x', '--scale-image', type = float, default = scale_image, \
                    help='Scaling factor applied at registration. Used for large images.')

log.add_argument(parser)

parser.add_argument('input_files', nargs = 2, default = input_filenames, \
                    help='TIFF image files. The first image is affine transformed.')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# turn on gpu
gpu_id = gpuimage.parse_gpu_argument(args)

# set arguments
input_filenames = args.input_files
register_all = args.register_all
reg_method = args.reg_method
opt_method = args.opt_method
truncate_frames = args.truncate_frames
scale_image = args.scale_image
channels = args.channels
output_json = args.output_json
scale_isometric = args.scale_isometric

init_flip = args.init_flip.lower()
init_rot = np.array(args.init_rot[::-1])
init_shift = np.array(args.init_shift[::-1])
post_shift = np.array(args.post_shift[::-1])

output_image_suffix = output_image_suffix.format(reg_method.lower())
output_json_suffix = output_json_suffix.format(reg_method.lower())

if args.output_image_file is None:
    output_image_filename = stack.with_suffix(input_filenames[1], output_image_suffix)
else:
    output_image_filename = args.output_image_file

if args.output_json_file is None:
    output_json_filename = stack.with_suffix(input_filenames[1], output_json_suffix)
else:
    output_json_filename = args.output_json_file

# read input TIFF
input_stacks = [stack.Stack(file) for file in input_filenames]
logger.info("Overlay image: {0}".format(input_filenames[0]))
logger.info("Overlay shape: {0}".format(input_stacks[0].image_array.shape))
logger.info("Overlay voxel: {0}".format(input_stacks[0].voxel_um))
logger.info("Background image: {0}".format(input_filenames[1]))
logger.info("Background shape: {0}".format(input_stacks[1].image_array.shape))
logger.info("Background voxel: {0}".format(input_stacks[1].voxel_um))

# pre-process the first image
logger.info("Pre-prosessing the first image, flip: {0}, rotation: {1}, shift {2}.".format(init_flip, init_rot, init_shift))
flip_x, flip_y, flip_z = [-1 if axis in init_flip else 1 for axis in 'xyz']
def preprocess (image, t_index, c_index):
    image = image[::flip_z, ::flip_y, ::flip_x]
    for axis, angle in zip(['x', 'y', 'z'], init_rot):
        if np.isclose(angle, 0.0) == False:
            image = gpuimage.rotate_by_axis(image, angle = angle, axis = axis, gpu_id = gpu_id)
    image = gpuimage.shift(image, init_shift, gpu_id = gpu_id)
    return image
input_stacks[0].apply_all(preprocess)

# scale images
if scale_isometric:
    pixel_um = min(input_stacks[0].voxel_um + input_stacks[1].voxel_um)
    logger.info("Scaling to achieve isometric pixelsize: {0}.".format(pixel_um))
    input_stacks[0].scale_by_pixelsize(pixel_um, gpu_id = gpu_id)
    input_stacks[1].scale_by_pixelsize(pixel_um, gpu_id = gpu_id)
else:
    scale_ratio = np.array(input_stacks[0].voxel_um) / np.array(input_stacks[1].voxel_um)
    if input_stacks[0].z_count == 1:
        scale_ratio[0] = 1
    logger.info("Scaling factor for the first image: {0}.".format(scale_ratio))
    input_stacks[0].scale_by_ratio(scale_ratio, gpu_id = gpu_id)

logger.info("The first image was shaped into: {0}.".format(input_stacks[0].image_array.shape))
logger.info("The second image was shaped into: {0}.".format(input_stacks[1].image_array.shape))

# max frames to be processed
if truncate_frames:
    max_frames = min(input_stacks[0].t_count, input_stacks[1].t_count)
else:
    max_frames = max(input_stacks[0].t_count, input_stacks[1].t_count)

# registration and preparing affine matrices
logger.info("Registration started. Reg: {0}. Opt: {1}.".format(reg_method, opt_method))
logger.info("Channels: {0}".format(channels))
logger.info("Scaling at registration: {0}.".format(scale_image))
affine_result_list = []
for index in progressbar(range(max_frames), redirect_stdout = True):
    # handle broadcasting
    t_indexes = [index % stack.t_count for stack in input_stacks]

    # registration
    if register_all or index == 0:
        images = [input_stacks[i].image_array[t_indexes[i], channels[i]] for i in range(len(input_stacks))]
        images[0] = gpuimage.resize(images[0], images[1].shape, centering = True)

        # scale image for registration
        if scale_image is not None:
            images = [gpuimage.scale(image, scale_image, gpu_id = gpu_id) for image in images]

        affine_result = register.register(images[1], images[0], init_shift = None, gpu_id = gpu_id, \
                                          reg_method = reg_method, opt_method = opt_method)

        # rescaling
        if scale_image is not None:
            affine_result['scale_image'] = scale_image
            affine_result['matrix'][0:3, 3] = affine_result['matrix'][0:3, 3] / scale_image

        affine_result['pre_flip'] = init_flip
        affine_result['pre_rotation'] = init_rot
        affine_result['pre_shift'] = init_shift
        affine_result['post_shift'] = post_shift

        # save result
        affine_result_list.append(copy.deepcopy(affine_result))

    else:
        affine_result_list.append(copy.deepcopy(affine_result_list[0]))

# affine transformation and overlay
output_stack = stack.Stack()
output_shape = list(input_stacks[1].image_array.shape)
output_shape[0] = max_frames
output_shape[1] = input_stacks[0].c_count + input_stacks[1].c_count
output_stack.alloc_zero_image(output_shape, dtype = np.float, \
                              voxel_um = input_stacks[1].voxel_um, \
                              finterval_sec = input_stacks[1].finterval_sec)

logger.info("Overlay started.")
logger.info("Post-processing shift: {0}".format(post_shift))
for index in progressbar(range(max_frames), redirect_stdout = True):
    # handle broadcasting
    t_indexes = [index % stack.t_count for stack in input_stacks]

    for c_index in range(input_stacks[0].c_count):
        image = input_stacks[0].image_array[t_indexes[0], c_index].astype(float)
        image = gpuimage.resize(image, output_shape[2:], centering = True)

        matrix = affine_result_list[index]['matrix']
        if len(matrix) == 4:
            matrix[0:3, 3] = matrix[0:3, 3] + post_shift
        else:
            matrix[0:2, 2] = matrix[0:2, 2] + post_shift[1:]

        image = gpuimage.affine_transform(image, affine_result_list[index]['matrix'], gpu_id = gpu_id)
        output_stack.image_array[index, c_index] = image

    for c_index in range(input_stacks[1].c_count):
        image = input_stacks[1].image_array[t_indexes[1], c_index].astype(float)
        output_stack.image_array[index, c_index + input_stacks[0].c_count] = image

# output image
logger.info("Saving image: {0}.".format(output_image_filename))
output_stack.save_ome_tiff(output_image_filename, dtype = np.float32)

# summarize the registration results and output
if output_json:
    params_dict = {'image_filenames': [Path(input_filename).name for input_filename in input_filenames],
                   'time_stamp': datetime.now().astimezone().isoformat(),
                   'voxelsize': input_stacks[1].voxel_um,
                   'channels': channels}
    summary_list = []
    for index in range(len(affine_result_list)):
        summary = {}
        summary['index'] = index
        summary['affine'] = affine_result_list[index]
        summary_list.append(summary)

    output_dict = {'parameters': params_dict, 'summary_list': summary_list}

    logger.info("Output JSON file: {0}".format(output_json_filename))
    with open(output_json_filename, 'w') as f:
        json.dump(output_dict, f, ensure_ascii = False, indent = 4, sort_keys = False, \
                  separators = (',', ': '), cls = NumpyEncoder)

