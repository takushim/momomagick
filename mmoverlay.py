#!/usr/bin/env python

import sys, argparse, time, json, copy
import numpy as np
from numpyencoder import NumpyEncoder
from mmtools import gpuimage, mmtiff, register

# default values
input_filenames = None
use_channels = None
init_shift = [0.0, 0.0, 0.0]
init_rotation = [0.0, 0.0, 0.0]
init_zoom = [1.0, 1.0, 1.0]
init_shear = [0.0, 0.0, 0.0]
post_shift = None
output_filename = None
output_suffix = '_over_{0}.tif' # overwritten by the registration method
output_json_filename = None
output_json_suffix = '.json' # overwritten by the registration method
truncate_frames = False
gpu_id = None
register_all = False
registering_method = 'Full'
registering_method_list = register.registering_methods
optimizing_method = "Powell"
optimizing_method_list = register.optimizing_methods

# parse arguments
parser = argparse.ArgumentParser(description='Overlay two time-lapse images after registration', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output TIFF file ([basename]{0} by default)'.format(output_suffix.format('[regmethod]')))

parser.add_argument('-j', '--output-json-file', default = output_json_filename, \
                    help='output JSON file name ([output_basename]{0} by default)'.format(output_json_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-a', '--register-all', action = 'store_true', \
                    help='Perform registration for each pair of images')

parser.add_argument('-n', '--truncate-frames', action = 'store_true', \
                    help='Disable broadcasting and truncate frames')

parser.add_argument('-e', '--registering-method', type = str, default = registering_method, \
                    choices = registering_method_list, \
                    help='Method used for registration')

parser.add_argument('-t', '--optimizing-method', type = str, default = optimizing_method, \
                    choices = optimizing_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-S', '--init-shift', nargs = 3, type = float, default = init_shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial shift of the overlay image (useful with "-e None")')

parser.add_argument('-R', '--init-rotation', nargs = 3, type = float, default = init_rotation, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial rotation of the overlay image')

parser.add_argument('-Z', '--init-zoom', nargs = 3, type = float, default = init_zoom, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial zoom of the overlay image')

parser.add_argument('-H', '--init-shear', nargs = 3, type = float, default = init_shear, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Initial shear of the overlay image')

parser.add_argument('-P', '--post-shift', nargs = 3, type = float, default = post_shift, \
                    metavar = ('X', 'Y', 'Z'), \
                    help='Post-registration shift of the overlay image (for adjustment")')

parser.add_argument('-c', '--use-channels', nargs = 2, type = int, default = use_channels, \
                    help='specify two channels used for registration')

parser.add_argument('input_files', nargs = 2, default = input_filenames, \
                    help='TIFF image files. The first image is affine transformed.')
args = parser.parse_args()

# set arguments
input_filenames = args.input_files
gpu_id = args.gpu_id
register_all = args.register_all
registering_method = args.registering_method
optimizing_method = args.optimizing_method
truncate_frames = args.truncate_frames

init_shift = np.array(args.init_shift[::-1])
init_rotation = np.array(args.init_rotation[::-1])
init_zoom = np.array(args.init_zoom[::-1])
init_shear = np.array(args.init_shear[::-1])

if args.post_shift is not None:
    post_shift = np.array(args.post_shift[::-1])

use_channels = args.use_channels
if use_channels is None:
    use_channels = [0, 0]

if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filenames[-1], output_suffix.format(registering_method.lower()))
else:
    output_filename = args.output_file

if args.output_json_file is None:
    output_json_filename = mmtiff.with_suffix(output_filename, output_json_suffix)
else:
    output_json_filename = args.output_json_file

# turn on GPU device
if gpu_id is not None:
    register.turn_on_gpu(gpu_id)

# read input TIFF
input_tiffs = [mmtiff.MMTiff(file) for file in input_filenames]
input_images = [tiff.as_list(list_channel = True) for tiff in input_tiffs]

# xy-scale images
pixelsize_list = [tiff.pixelsize_um for tiff in input_tiffs]
pixelsize_um = np.min(pixelsize_list)
xy_ratio_list = np.array(pixelsize_list) / pixelsize_um
for file_index, xy_ratio in enumerate(xy_ratio_list):
    if np.isclose(xy_ratio_list[file_index], 1.0) == False:
        print("XY-scaling image #{0} at ratio = {1}".format(file_index, xy_ratio))
        for index in range(input_tiffs[file_index].total_time):
            for channel in range(input_tiffs[file_index].total_channel):
                input_images[file_index][index][channel] = gpuimage.zoom(input_images[file_index][index][channel], \
                                                                         ratio = (1.0, xy_ratio, xy_ratio), gpu_id = gpu_id)

if np.isclose(xy_ratio_list[0], 1.0) == False:
    print("Rescaling the xy overlay offset:", init_shift)
    init_shift[1:] = init_shift[1:] * xy_ratio

# z-scale images
z_step_list = [tiff.z_step_um for tiff in input_tiffs]
z_step_um = np.min(z_step_list)
z_ratio_list = np.array(z_step_list) / z_step_um
for file_index, z_ratio in enumerate(z_ratio_list):
    if np.isclose(z_ratio_list[file_index], 1.0) == False:
        print("Z-scaling image #{0} at ratio = {1}".format(file_index, z_ratio))
        for index in range(input_tiffs[file_index].total_time):
            for channel in range(input_tiffs[file_index].total_channel):
                input_images[file_index][index][channel] = gpuimage.z_zoom(input_images[file_index][index][channel], \
                                                                           ratio = z_ratio, gpu_id = gpu_id)

if np.isclose(z_ratio_list[0], 1.0) == False:
    print("Rescaling the z overlay offset:", init_shift)
    init_shift[0] = init_shift[0] * z_ratio

# show image sizes
for index in range(len(input_images)):
    print("Image:", index, ", shape:", input_images[index][0][0].shape)


# registration and preparing affine matrices
print("Start registration:", time.ctime())
print("Registering Method:", registering_method)
print("Optimizing Method:", optimizing_method)
print("Channels:", use_channels)

affine_result_list = []
if truncate_frames:
    max_frames = min([tiff.total_time for tiff in input_tiffs])
else:
    max_frames = max([tiff.total_time for tiff in input_tiffs])

for index in range(max_frames):
    # handle broadcasting
    over_index = index % input_tiffs[0].total_time
    ref_index = index % input_tiffs[1].total_time

    # registration
    if register_all or index == 0:
        print("Frame:", index)
        over_image = input_images[0][over_index][use_channels[0]]
        ref_image = input_images[1][ref_index][use_channels[1]]

        # resize the overlaying image
        if over_image.shape != ref_image.shape:
            over_image = gpuimage.resize(over_image, ref_image.shape, center = True)

        init_params = register.params_dict_3d(shift = init_shift, rotation = init_rotation, \
                                              zoom = init_zoom, shear = init_shear)
        affine_result = register.register(ref_image, over_image, init_params = init_params, \
                                          gpu_id = gpu_id, \
                                          reg_method = registering_method, \
                                          opt_method = optimizing_method)
        print(affine_result['results'].message)

        affine_result['init_params'] = init_params
        affine_matrix = affine_result['matrix']

        print("Matrix:")
        print(affine_matrix)

        # interpret the affine matrix
        decomposed_matrix = register.decompose_matrix(affine_matrix)
        print("Shift:", decomposed_matrix['shift'])
        print("Rotation:", decomposed_matrix['rotation_angles'])
        print("Zoom:", decomposed_matrix['zoom'])
        print("Shear:", decomposed_matrix['shear'])

    # save result
    affine_result_list.append(copy.deepcopy(affine_result))

print("End registration:", time.ctime())
print(".")

# affine transformation and overlay
print("Start overlay:", time.ctime())
print("Post-processing shift:", post_shift)
print("Frame:", end = ' ')
output_image_list = []
ref_shape = input_images[1][0][0].shape
for index in range(len(affine_result_list)):
    # handle broadcasting
    over_index = index % input_tiffs[0].total_time
    ref_index = index % input_tiffs[1].total_time

    # prepare output images
    image_list = []

    # post-processing offset
    affine_matrix = affine_result_list[index]['matrix']
    if post_shift is not None:
        affine_matrix = register.offset_matrix(affine_matrix, post_shift)
        affine_result_list[index]['post_shift'] = post_shift
        affine_result_list[index]['matrix_offset'] = affine_matrix

    # overlay images
    for channel in range(input_tiffs[0].total_channel):
        image = input_images[0][over_index][channel]
        if image.shape != ref_shape:
            image = gpuimage.resize(image, ref_shape, center = True)
        image = gpuimage.affine_transform(image, affine_matrix, gpu_id = gpu_id)
        image_list.append(image)

    # reference images
    for channel in range(input_tiffs[1].total_channel):
        image = input_images[1][ref_index][channel]
        image_list.append(image)

    print(index, end = ' ', flush = True)
    output_image_list.append(image_list)

print(".")
print("End overlay:", time.ctime())

# summarize the registration results and output
params_dict = {'image_filename': input_filenames,
               'time_stamp': time.strftime("%a %d %b %H:%M:%S %Z %Y")}
summary_list = []
for index in range(len(affine_result_list)):
    summary = {}
    summary['index'] = index
    summary['affine'] = affine_result_list[index]
    summary_list.append(summary)

output_dict = {'parameters': params_dict, 'summary_list': summary_list}

print("Output JSON file:", output_json_filename)
with open(output_json_filename, 'w') as f:
    json.dump(output_dict, f, \
              ensure_ascii = False, indent = 4, sort_keys = False, \
              separators = (',', ': '), cls = NumpyEncoder)

# output image
print("Output image:", output_filename)
output_image = np.array(output_image_list).swapaxes(1, 2)
mmtiff.save_image(output_filename, output_image, imagej = True, \
                  xy_pixel_um = pixelsize_um, \
                  z_step_um = z_step_um)
