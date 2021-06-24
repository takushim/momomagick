#!/usr/bin/env python

import sys, argparse, json, time
import numpy as np
from pathlib import Path
from numpyencoder import NumpyEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess
from mmtools import mmtiff, regist

# defaults
input_filename = None
output_txt_filename = None
output_txt_suffix = '_affine.json'
ref_filename = None
use_channel = 0
output_aligned_image = False
aligned_image_filename = None
aligned_image_suffix = '_reg.tif'
transport_only = False
gpu_id = None

parser = argparse.ArgumentParser(description='Register time-lapse images using affine matrix and optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-txt-file', default = output_txt_filename, \
                    help='output JSON file name ([basename]{0} if not specified)'.format(output_txt_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

parser.add_argument('-t', '--transport-only', action = 'store_true', \
                    help='Optimize for parallel transport only')

parser.add_argument('-A', '--output-aligned-image', action = 'store_true', \
                    help='output aligned images')

parser.add_argument('-a', '--aligned-image-file', default = aligned_image_filename, \
                    help='filename to output images ([basename]{0} if not specified)'.format(aligned_image_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
ref_filename = args.ref_image
use_channel = args.use_channel
gpu_id = args.gpu_id
transport_only = args.transport_only
output_aligned_image = args.output_aligned_image

if args.output_txt_file is None:
    output_txt_filename = mmtiff.with_suffix(input_filename, output_txt_suffix)
else:
    output_txt_filename = args.output_txt_file

if args.aligned_image_file is None:
    aligned_image_filename = mmtiff.with_suffix(input_filename, aligned_image_suffix)
else:
    aligned_image_filename = args.aligned_image_file

# turn on GPU device
if gpu_id is not None:
    regist.turn_on_gpu(gpu_id)

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

# calculate POCs for pre-registration
poc_result_list = []
poc_register = regist.Poc(ref_image, gpu_id = gpu_id)
print("Pre-registrating using phase-only-correlation.")
for index in range(len(input_images)):
    poc_result = poc_register.regist(input_images[index])
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

# optimization for each affine matrix
# note: input = matrix * output + offset
affine_result_list = []
output_image_list = []
affine_register = regist.Affine(ref_image, gpu_id = gpu_id)
for index in range(len(input_images)):
    print("Starting optimization:", index)

    if input_tiff.total_zstack == 1:
        init_shift = init_shift_list[index]['shift'][1:]
        input_image = input_images[index][0]
    else:
        init_shift = init_shift_list[index]['shift']
        input_image = input_images[index]
    print("Initial shift:", init_shift)

    affine_result = affine_register.regist(input_image, init_shift, transport_only = transport_only)
    final_matrix = affine_result['matrix']

    if output_aligned_image:
        output_image = regist.affine_transform(input_image, final_matrix, gpu_id)
        if input_tiff.total_zstack == 1:
            output_image = output_image[np.newaxis, np.newaxis]
        else:
            output_image = output_image[:, np.newaxis]
        output_image_list.append(output_image)

    print(affine_result['results'].message)
    print("Matrix:")
    print(final_matrix)

    # interpret the affine matrix
    decomposed_matrix = regist.decompose_matrix(final_matrix)
    affine_result['decomposed'] = decomposed_matrix
    print("Transport:", decomposed_matrix['transport'])
    print("Rotation:", decomposed_matrix['rotation_angles'])
    print("Zoom:", decomposed_matrix['zoom'])
    print("Shear:", decomposed_matrix['shear'])

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
    input_tiff.save_image_ome(aligned_image_filename, np.array(output_image_list))
