#!/usr/bin/env python

import argparse, json
import numpy as np
from datetime import datetime
from pathlib import Path
from progressbar import progressbar
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import gpuimage, stack, register, log, npencode

# defaults
input_filename = None
input_channel = 0
output_json_filename = None
output_json_suffix = '_reg.json' # overwritten
output_image_filename = None
output_image_suffix = '_reg.tif' # overwritten
ref_filename = None
ref_channel = 0
reg_method = 'Full'
reg_method_list = register.registering_methods
opt_method = "Powell"
opt_method_list = register.optimizing_methods
reg_area = None

parser = argparse.ArgumentParser(description='Register time-lapse images using affine matrix and optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-image-file', default = output_image_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_image_suffix))

gpuimage.add_gpu_argument(parser)

parser.add_argument('-J', '--output-json', action = 'store_true', \
                    help='Enable output of the json file')

parser.add_argument('-j', '--output-json-file', default = output_json_filename, \
                    help='filename of output json ([basename]{0} if not specified)'.format(output_json_suffix))

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-n', '--ref-channel', type = int, default = ref_channel, \
                    help='specify the channel of the reference image to process')

parser.add_argument('-c', '--input-channel', type = int, default = input_channel, \
                    help='specify the channel of the input image to process')

parser.add_argument('-R', '--reg-area', type = int, nargs = 4, default = reg_area, \
                   metavar = ('X', 'Y', 'W', "H"),
                   help='Register using the specified area.')

parser.add_argument('-e', '--reg-method', type = str, default = reg_method, \
                    choices = reg_method_list, \
                    help='Method used for registration')

parser.add_argument('-t', '--opt-method', type = str, default = opt_method, \
                    choices = opt_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-s', '--scale-isometric', action = 'store_true', \
                    help='Scale Z-axis to achieve isotrophic voxels')

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
input_channel = args.input_channel
ref_filename = args.ref_image
ref_channel = args.ref_channel
output_json = args.output_json
reg_method = args.reg_method
opt_method = args.opt_method
reg_area = args.reg_area
scale_isometric = args.scale_isometric

if args.output_image_file is None:
    output_image_filename = stack.with_suffix(input_filename, output_image_suffix)
else:
    output_image_filename = args.output_image_file

if args.output_json_file is None:
    output_json_filename = stack.with_suffix(input_filename, output_json_suffix)
else:
    output_json_filename = args.output_json_file

# read input image
input_stack = stack.Stack(input_filename, keep_s_axis = False)
if scale_isometric:
    logger.info("Scaling image to be isometric. Pixel-size: {0}.".format(min(input_stack.voxel_um)))
    input_stack.scale_isometric(gpu_id = gpu_id)

# setting the z slice to switch between the 3D and 2D modes
if input_stack.z_count > 1:
    z_slice = slice(0, input_stack.z_count, 1)
    image_dim = 3
else:
    z_slice = 0
    image_dim = 2

# read reference image
if ref_filename is None:
    ref_image = input_stack.image_array[0, input_channel].copy()
else:
    logger.info("Using a reference image: {0}. Channel {1}.".format(ref_filename, ref_channel))
    ref_stack = stack.Stack(ref_filename, keep_s_axis = False)
    ref_image = ref_stack.image_array[0, ref_channel]
    if np.allclose(input_stack.voxel_um, ref_stack.voxel_um, atol = 1e-2) == False:
        logger.warning("Pixel sizes are different. Input: {0}. Ref: {1}.".format(input_stack.voxel_um, ref_stack.voxel_um))

# prepare slices to crop areas used for registration
if reg_area is None:
    reg_area = [0, 0, input_stack.width, input_stack.height]
reg_slice_x = slice(reg_area[0], reg_area[0] + reg_area[2], 1)
reg_slice_y = slice(reg_area[1], reg_area[1] + reg_area[3], 1)
logger.info("Using area: {0}, channel: {1}.".format(reg_area, input_channel))

# calculate POCs for pre-registration
poc_result_list = []

poc_register = register.Poc(ref_image[z_slice, reg_slice_y, reg_slice_x], gpu_id = gpu_id)
if reg_method == 'None':
    logger.info("Skipping Pre-registration.")
    poc_result_list = [{'shift': [0.0, 0.0, 0.0], 'corr': 1.0}] * input_stack.t_count
elif reg_method == 'INTPOC':
    logger.info("Pre-registrating using coarse phase-only-correlation.")
    for index in progressbar(range(input_stack.t_count), redirect_stdout = True):
        poc_result = poc_register.register(input_stack.image_array[index, input_channel, z_slice, reg_slice_y, reg_slice_x])
        poc_result_list.append(poc_result)
else:
    logger.info("Pre-registrating using subpixel phase-only-correlation.")
    for index in progressbar(range(input_stack.t_count), redirect_stdout = True):
        poc_result = poc_register.register_subpixel(input_stack.image_array[index, input_channel, z_slice, reg_slice_y, reg_slice_x])
        poc_result_list.append(poc_result)

poc_register = None

# lowess filter to exclude outliers
logger.info("Applying a lowess filter to the POC results.")
lowess_list = []
for index in range(image_dim):
    values = np.array([result['shift'][index] for result in poc_result_list])
    values = lowess(values, np.arange(input_stack.t_count), frac = 0.1, return_sorted = False)
    values = values - values[0]
    lowess_list.append(values)

lowess_array = np.array(lowess_list).swapaxes(0, 1)
for index in range(input_stack.t_count):
    poc_result_list[index]['poc_shift'] = list(lowess_array[index])

# optimization for each affine matrix
# note: input = matrix * output + offset
affine_result_list = []
output_image_list = []
affine_register = register.Affine(ref_image[z_slice, reg_slice_y, reg_slice_x], gpu_id = gpu_id)

logger.info("Optimization started. Reg: {0}. Opt: {1}.".format(reg_method, opt_method))
for index in progressbar(range(input_stack.t_count), redirect_stdout = True):
    init_shift = poc_result_list[index]['poc_shift']
    input_image = input_stack.image_array[index, input_channel, z_slice, reg_slice_y, reg_slice_x]
    affine_result = affine_register.register(input_image, init_shift = init_shift, \
                                             opt_method = opt_method, reg_method = reg_method)
    affine_result_list.append(affine_result)

affine_register = None

# output the aligned image.
logger.info("Preparing aligned image: {0}.".format(output_image_filename))
if input_stack.z_count > 1:
    def affine_transform (image, t_index, c_index):
        matrix = affine_result_list[t_index]['matrix']
        return gpuimage.affine_transform(image, matrix, gpu_id = gpu_id)
else:
    def affine_transform (image, t_index, c_index):
        matrix = affine_result_list[t_index]['matrix']
        return gpuimage.affine_transform(image[0], matrix, gpu_id = gpu_id)[np.newaxis]

input_stack.apply_all(affine_transform, progress = True)
input_stack.save_ome_tiff(output_image_filename)

# summarize and output the results
if output_json:
    params_dict = {'image_filename': Path(input_filename).name,
                  'time_stamp': datetime.now().astimezone().isoformat(),
                  'voxelsize': input_stack.voxel_um,
                  'input_channel': input_channel,
                  'reg_area': reg_area}

    if ref_filename is not None:
        params_dict['ref_filename'] = ref_filename
        params_dict['ref_channel'] = ref_channel

    summary_list = []
    for index in range(input_stack.t_count):
        summary = {}
        summary['index'] = index
        summary['poc'] = poc_result_list[index]
        summary['affine'] = affine_result_list[index]
        summary_list.append(summary)

    output_dict = {'parameters': params_dict, 'summary_list': summary_list}

    logger.info("Output JSON file: {0}.".format(output_json_filename))
    with open(output_json_filename, 'w') as file:
        json.dump(output_dict, file, ensure_ascii = False, indent = 4, sort_keys = False, \
                  separators = (', ', ': '), cls = npencode.Encoder)

