#!/usr/bin/env python

import sys, argparse
import numpy as np
import pandas as pd
from mmtools import mmtiff, regist

# defaults
input_filename = None
output_tsv_filename = None
output_tsv_suffix = '_affine.txt'
ref_filename = None
use_channel = 0
output_aligned_image = False
aligned_image_filename = None
aligned_image_suffix = '_reg.tif'
transport_only = False
gpu_id = None

parser = argparse.ArgumentParser(description='Register time-lapse images using affine matrix and optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-tsv-file', default = output_tsv_filename, \
                    help='output TSV file name ([basename]{0} if not specified)'.format(output_tsv_suffix))

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

if args.output_tsv_file is None:
    output_tsv_filename = mmtiff.with_suffix(input_filename, output_tsv_suffix)
else:
    output_tsv_filename = args.output_tsv_file

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

# optimization for each affine matrix
# note: input = matrix * output + offset
init_shift_list = [{'shift': poc_result['shift']} for poc_result in poc_result_list]
affine_result_list = []
output_image_list = []
affine_register = regist.Affine(ref_image, gpu_id = gpu_id)
for index in range(len(input_images)):
    print("Starting optimization:", index)

    init_shift = init_shift_list[index]['shift']
    if input_tiff.total_zstack == 1:
        input_image = input_images[index][0]
    else:
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

    print(affine_result['results'].message, "Matrix:")
    print(final_matrix)

    affine_result_list.append(affine_result)
    print(".")

# free gpu memory
affine_register = None

# summarize the results
summary_list = []
for index in range(len(input_images)):
    summary = {'index': index}
    summary.update({"poc_{0}".format(x): y for x, y in zip(['x', 'y', 'z'], poc_result_list[index]['shift'][::-1])})
    summary.update({"poc_corr": poc_result_list[index]['corr']})
    summary.update({"init_{0}".format(x): y for x, y in zip(['x', 'y', 'z'], init_shift_list[index]['shift'][::-1])})
    summary.update({"aff_stat": affine_result_list[index]['results'].status})
    summary.update({"aff_mat": ",".join([str(x) for x in affine_result_list[index]['results'].x])})
    summary_list.append(summary)

# open tsv file and write header
print("Output TSV:", output_tsv_filename)
shift_table = pd.DataFrame(summary_list)
shift_table.to_csv(output_tsv_filename, sep = '\t', index = False)

# output images
if output_aligned_image:
    print("Output image:", aligned_image_filename)
    input_tiff.save_image_ome(aligned_image_filename, np.array(output_image_list))
