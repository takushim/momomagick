#!/usr/bin/env python

import sys, argparse, tifffile
import numpy as np
from pathlib import Path
from mmtools import mmtiff, register, lucy, gpuimage

# default values
input_filename = None
output_filename = None
output_suffix = '_fusion.tif'
main_channel = 0
sub_channel = 1
sub_rotation = 0
gpu_id = None
registering_method = 'Full'
registering_method_list = register.registering_methods
optimizing_method = "Powell"
optimizing_method_list = register.optimizing_methods
psf_folder = Path(__file__).parent.joinpath('psf')
psf_filename = 'diSPIM.tif'
iterations = 10

# parse arguments
parser = argparse.ArgumentParser(description='Fusion two diSPIM images and deconvolve them.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default=output_filename, \
                    help='Output TIFF file ([basename0]{0} by default)'.format(output_suffix))

parser.add_argument('-m', '--main-channel', type = int, default = main_channel, \
                    help='Channel to map the XY plane of output')

parser.add_argument('-s', '--sub-channel', type = int, default = sub_channel, \
                    help='Explicitly specify channel to map the XZ plane')

parser.add_argument('-r', '--sub-rotation', type = int, 
                    help='Angle to rotation the sub-channel around the Y axis')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-e', '--registering-method', type = str, default = registering_method, \
                    choices = registering_method_list, \
                    help='Method used for image registration')

parser.add_argument('-t', '--optimizing-method', type = str, default = optimizing_method, \
                    choices = optimizing_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-i', '--iterations', default = iterations, \
                    help='number of iterations')

parser.add_argument('input_file', default = input_filename, \
                    help='Input dual-view TIFF file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
main_channel = args.main_channel
sub_channel = args.sub_channel
sub_rotation = args.sub_rotation
gpu_id = args.gpu_id
registering_method = args.registering_method
optimizing_method = args.optimizing_method
psf_filename = args.psf_image
iterations = args.iterations

if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
register.turn_on_gpu(gpu_id)

# read input TIFF
input_tiff = mmtiff.MMTiff(input_filename)
input_image_list = input_tiff.as_list(list_channel = True)

# load psf image
if Path(psf_filename).exists():
    print("Read PSF image in the current folder:", psf_filename)
    psf_image = tifffile.imread(psf_filename)
else:
    psf_path = Path(psf_folder).joinpath(psf_filename)
    if psf_path.exists():
        print("Read PSF image in the system folder:", str(psf_path))
        psf_image = tifffile.imread(str(psf_path))
    else:
        raise Exception('PSF file {0} not found'.format(psf_filename))

# prepare a rotated psf image
if sub_rotation != 0:
    print("Preparing a psf image for the sub-channel rotating by:", sub_rotation)
    psf_image_sub = gpuimage.z_rotate(psf_image, angle = sub_rotation, gpu_id = gpu_id)

# finding the other channel
channel_set = set(np.arange(input_tiff.total_channel)) - {main_channel}
if sub_channel not in channel_set:
    sub_channel = min(channel_set)
    print("Automatically selecting the sub-channel:", sub_channel)

# scaling along the z-axis to achieve isometric voxels
z_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um

# rotate, register and deconvolve
output_image_list = []
affine_result_list = []
deconvolver = lucy.Lucy([psf_image, psf_image_sub], gpu_id = gpu_id)
for index in range(input_tiff.total_time):
    # images
    main_image = input_image_list[index][main_channel]
    sub_image = input_image_list[index][sub_channel]

    main_image = gpuimage.z_zoom(main_image, ratio = z_ratio, gpu_id = gpu_id)
    sub_image = gpuimage.z_zoom(sub_image, ratio = z_ratio, gpu_id = gpu_id)
    if sub_rotation != 0:
        print("Rotating sub-channel by:", sub_rotation)
        sub_image = gpuimage.z_rotate(sub_image, angle = sub_rotation, gpu_id = gpu_id)

    sub_image = gpuimage.resize(sub_image, main_image.shape, center = True)

    print("Registering Method:", registering_method)
    print("Optimizing Method:", optimizing_method)
    affine_result = register.register(main_image, sub_image, gpu_id = gpu_id, \
             opt_method = optimizing_method, reg_method = registering_method)
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

    # fuse two channels
    sub_image = gpuimage.affine_transform(sub_image, affine_matrix, gpu_id = gpu_id)
    dual_image = (main_image.astype(float) + sub_image.astype(float)) / 2

    # deconvolution
    dual_image = deconvolver.deconvolve(dual_image, iterations)

    # store the image to the list
    output_image_list.append(dual_image)

# shape output into the TZCYX order
output_image = np.array(output_image_list)[:, :, np.newaxis]
if (input_tiff.dtype.kind == 'i' or input_tiff.dtype.kind == 'u') and \
        np.max(output_image) <= np.iinfo(input_tiff.dtype).max:
    output_image = mmtiff.float_to_int(output_image, input_tiff.dtype)
else:
    output_image = output_image.astype(np.float32)

# output in the ImageJ format, dimensions should be in TZCYX order
print("Output image:", output_filename)
input_tiff.z_step_um = input_tiff.pixelsize_um
input_tiff.save_image(output_filename, output_image)