#!/usr/bin/env python

import sys, argparse
import numpy as np
from pathlib import Path
from mmtools import mmtiff, register

# default values
input_filename = None
overlay_filename = None
output_filename = None
output_suffix = '_impose.tif'
input_channel = 0
overlay_channel = 0
gpu_id = None
register_all = False
registering_method = 'Full'
registering_method_list = register.registering_methods
optimizing_method = "Powell"
optimizing_method_list = register.optimizing_methods

# parse arguments
parser = argparse.ArgumentParser(description='Impose two series of time-lapse/still images', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='output TIFF file ([basename0]{0} by default)'.format(output_suffix))

parser.add_argument('-l', '--overlay-channel', type = int, default = overlay_channel, \
                    help='channel in the overlaying image used for registration')

parser.add_argument('-n', '--input-channel', type = int, default = input_channel, \
                    help='channel in the input image used for registration')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-a', '--register-all', action = 'store_true', \
                    help='Perform registration for each pair of images')

parser.add_argument('-e', '--registering-method', type = str, default = registering_method, \
                    choices = registering_method_list, \
                    help='Method used for registration')

parser.add_argument('-p', '--optimizing-method', type = str, default = optimizing_method, \
                    choices = optimizing_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('overlay_file', default = overlay_filename, \
                    help='TIFF file used for overlay (broadcasted if necessary)')

parser.add_argument('input_file', default = input_filename, \
                    help='TIFF file (original image)')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
overlay_filename = args.overlay_file
input_channel = args.input_channel
overlay_channel = args.overlay_channel
gpu_id = args.gpu_id
register_all = args.register_all
registering_method = args.registering_method
optimizing_method = args.optimizing_method
if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
if gpu_id is not None:
    register.turn_on_gpu(gpu_id)

# read input TIFF
input_tiff = mmtiff.MMTiff(input_filename)
input_image_list = input_tiff.as_list(list_channel = True)

# read overlay TIFF
overlay_tiff = mmtiff.MMTiff(overlay_filename)
overlay_image_list = overlay_tiff.as_list(list_channel = True)

# registration
if input_tiff.total_zstack == 1 or overlay_tiff.total_zstack == 1:
    print("No z-scaling for 2D images")
    z_scaling = False
elif np.isclose(input_tiff.z_step_um, overlay_tiff.z_step_um):
    print("No z-scaling since z-step sizes are close")
    z_scaling = False
else:
    z_scaling = True
    if input_tiff.z_step_um > overlay_tiff.z_step_um:
        z_scale_input = True
        z_ratio = input_tiff.z_step_um / overlay_tiff.z_step_um
        print("Z-scaling for input images:", z_ratio)
    else:
        z_scale_input = False
        z_ratio = overlay_tiff.z_step_um / input_tiff.z_step_um
        print("Z-scaling for overlay images:", z_ratio)

output_image_list = []
affine_result_list = []
for index in range(input_tiff.total_time):
    # handle broadcasting
    overlay_index = index % overlay_tiff.total_time

    if register_all or index == 0:
        print("Registering Method:", registering_method)
        print("Optimizing Method:", optimizing_method)
        affine_result = register.register(input_image_list[index][input_channel], \
                                          overlay_image_list[overlay_index][overlay_channel], \
                                          gpu_id = gpu_id, reg_method = registering_method, \
                                          opt_method = optimizing_method)
        affine_matrix = affine_result['matrix']

        print(affine_result['results'].message)
        print("Matrix:")
        print(affine_matrix)

        # interpret the affine matrix
        decomposed_matrix = register.decompose_matrix(affine_matrix)
        print("Transport:", decomposed_matrix['transport'])
        print("Rotation:", decomposed_matrix['rotation_angles'])
        print("Zoom:", decomposed_matrix['zoom'])
        print("Shear:", decomposed_matrix['shear'])
        print(".")

        # save result
        affine_result_list.append(affine_result)
    else:
        affine_matrix = affine_result_list[0]['matrix']

    # prepare output images
    if z_scaling and z_scale_input:
        image_list = []
        for channel in range(input_tiff.total_channel):
            image = register.z_zoom(input_image_list[index][channel], z_ratio, gpu_id = gpu_id)
            image_list.append(image)
    else:
        image_list = input_image_list[index]

    for channel in range(overlay_tiff.total_channel):
        if overlay_tiff.total_zstack == 1:
            image = register.affine_transform(overlay_image_list[overlay_index][channel][0], affine_matrix, gpu_id = gpu_id)
            image = image[np.newaxis]
        else:
            image = overlay_image_list[overlay_index][channel]
            # z_scaling
            if z_scaling and (z_scale_input is False):
                image = register.z_zoom(image, z_ratio, gpu_id = gpu_id)
            image = register.affine_transform(image, affine_matrix, gpu_id = gpu_id)
        image_list.append(image)

    output_image_list.append(image_list)

# output image
print("Output image:", output_filename)
input_tiff.save_image(output_filename, np.array(output_image_list).swapaxes(1, 2))
