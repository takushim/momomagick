#!/usr/bin/env python

import sys, argparse, json, time
import numpy as np
from pathlib import Path
from numpyencoder import NumpyEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import mmtiff, register, gpuimage

# defaults
input_filename = None
output_txt_filename = None
output_txt_suffix = '_{0}.json' # overwritten
ref_filename = None
use_channel = 0
output_aligned_image = False
aligned_image_filename = None
aligned_image_suffix = '_{0}.tif' # overwritten
gpu_id = None
registering_method = 'Full'
registering_method_list = register.registering_methods
optimizing_method = "Powell"
optimizing_method_list = register.optimizing_methods
use_area = None
preset_area_index = None
preset_areas = mmtiff.preset_areas

parser = argparse.ArgumentParser(description='Register time-lapse images using affine matrix and optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-txt-file', default = output_txt_filename, \
                    help='output JSON file name ([basename]{0} if not specified)'.format(output_txt_suffix.format('[regmethod]')))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

group = parser.add_mutually_exclusive_group()
group.add_argument('-P', '--preset-area-index', type = int, default = preset_area_index, \
                   help='Register using the preset area. ' + \
                        ' '.join(["Area {0}: X {1} Y {2} W {3} H {4}.".format(i, *preset_areas[i]) \
                                  for i in range(len(preset_areas))]))
group.add_argument('-R', '--use-area', type = int, nargs = 4, default = use_area, \
                   metavar = ('X', 'Y', 'W', "H"),
                   help='Register using the specified area.')

parser.add_argument('-e', '--registering-method', type = str, default = registering_method, \
                    choices = registering_method_list, \
                    help='Method used for registration')

parser.add_argument('-t', '--optimizing-method', type = str, default = optimizing_method, \
                    choices = optimizing_method_list, \
                    help='Method to optimize the affine matrices')

parser.add_argument('-A', '--output-aligned-image', action = 'store_true', \
                    help='output aligned images')

parser.add_argument('-a', '--aligned-image-file', default = aligned_image_filename, \
                    help='filename to output images ([basename]{0} if not specified)'.format(aligned_image_suffix.format('[regmethod]')))

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
ref_filename = args.ref_image
use_channel = args.use_channel
gpu_id = args.gpu_id
output_aligned_image = args.output_aligned_image
registering_method = args.registering_method
optimizing_method = args.optimizing_method

output_txt_suffix = output_txt_suffix.format(registering_method.lower())
aligned_image_suffix = aligned_image_suffix.format(registering_method.lower())

if args.output_txt_file is None:
    output_txt_filename = mmtiff.with_suffix(input_filename, output_txt_suffix)
else:
    output_txt_filename = args.output_txt_file

if args.aligned_image_file is None:
    aligned_image_filename = mmtiff.with_suffix(input_filename, aligned_image_suffix)
else:
    aligned_image_filename = args.aligned_image_file

if args.use_area is not None:
    use_area = args.use_area
elif args.preset_area_index is not None:
    use_area = preset_areas[args.preset_area_index]

# turn on GPU device
if gpu_id is not None:
    register.turn_on_gpu(gpu_id)

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')
input_images = input_tiff.as_list(channel = use_channel, drop = True)

# read reference image
if ref_filename is None:
    ref_image = input_images[0].copy()
else:
    ref_tiff = mmtiff.MMTiff(ref_filename)
    if ref_tiff.colored:
        raise Exception('Reference image: color reference image not accepted.')
    ref_image = ref_tiff.as_list(channel = use_channel, drop = True)[0]

# prepare slices to crop areas used for registration
if use_area is None:
    use_area = [0, 0, input_tiff.width, input_tiff.height]

reg_slice_x, reg_slice_y = mmtiff.area_to_slice(use_area)
print("Using X slice:", reg_slice_x)
print("Using Y slice:", reg_slice_y)

# calculate POCs for pre-registration
poc_result_list = []
print("Pre-registrating using phase-only-correlation.")
poc_register = register.Poc(ref_image[..., reg_slice_y, reg_slice_x], gpu_id = gpu_id)
for index in range(len(input_images)):
    poc_result = poc_register.register(input_images[index][..., reg_slice_y, reg_slice_x])
    poc_result_list.append(poc_result)

# free gpu memory
poc_register = None

# losess filter to exclude outliers
values_list = []
for i in range(len(ref_image.shape)):
    values = np.array([x['shift'][i] for x in poc_result_list])
    values = lowess(values, np.arange(len(input_images)), frac = 0.1, return_sorted = False)
    values = values - values[0]
    values_list.append(values)
values_list = np.array(values_list)
init_shift_list = [{'shift': values_list[:, i]} for i in range(len(input_images))]
print("Shift in the last plane:", init_shift_list[-1]['shift'])

# optimization for each affine matrix
# note: input = matrix * output + offset
affine_result_list = []
output_image_list = []
affine_register = register.Affine(ref_image[..., reg_slice_y, reg_slice_x], gpu_id = gpu_id)
for index in range(len(input_images)):
#for index in range(1):
    print("Starting optimization:", index)
    print("Registering Method:", registering_method)
    print("Optimizing Method:", optimizing_method)

    start_time = time.perf_counter()

    if input_tiff.total_zstack == 1:
        init_shift = init_shift_list[index]['shift'][1:]
        input_image = input_images[index][0]
    else:
        init_shift = init_shift_list[index]['shift']
        input_image = input_images[index]
    print("Initial shift:", init_shift)

    affine_result = affine_register.register(input_image[..., reg_slice_y, reg_slice_x], init_shift, \
                                             opt_method = optimizing_method, reg_method = registering_method)
    final_matrix = affine_result['matrix']

    if output_aligned_image:
        output_image = gpuimage.affine_transform(input_image, final_matrix, gpu_id = gpu_id)
        if input_tiff.total_zstack == 1:
            output_image = output_image[np.newaxis, np.newaxis]
        else:
            output_image = output_image[:, np.newaxis]
        output_image_list.append(output_image)

    print(affine_result['results'].message)
    print("Matrix:")
    print(final_matrix)

    # interpret the affine matrix
    decomposed_matrix = register.decompose_matrix(final_matrix)
    affine_result['decomposed'] = decomposed_matrix
    print("Transport:", decomposed_matrix['transport'])
    print("Rotation:", decomposed_matrix['rotation_angles'])
    print("Zoom:", decomposed_matrix['zoom'])
    print("Shear:", decomposed_matrix['shear'])

    # Output time required for calculation
    elapsed_time = time.perf_counter() - start_time
    print("Elapsed time:", elapsed_time)
    affine_result['elapsed_time'] = elapsed_time

    affine_result_list.append(affine_result)
    print(".")

# free gpu memory
affine_register = None

# summarize the results
params_dict = {'image_filename': Path(input_filename).name,
               'time_stamp': time.strftime("%a %d %b %H:%M:%S %Z %Y")}
summary_list = []
for index in range(len(input_images)):
    summary = {}
    summary['index'] = index
    summary['poc'] = poc_result_list[index]
    summary['affine'] = affine_result_list[index]
    summary_list.append(summary)

output_dict = {'parameters': params_dict, 'summary_list': summary_list}

# open tsv file and write header
print("Output JSON file:", output_txt_filename)
with open(output_txt_filename, 'w') as f:
    json.dump(output_dict, f, \
              ensure_ascii = False, indent = 4, sort_keys = False, \
              separators = (',', ': '), cls = NumpyEncoder)

# output images
if output_aligned_image:
    print("Output image:", aligned_image_filename)
    input_tiff.save_image(aligned_image_filename, np.array(output_image_list))
