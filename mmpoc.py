#!/usr/bin/env python

import sys, argparse
import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from mmtools import mmtiff

# defaults
input_filename = None
output_filename = 'align.txt'
ref_filename = None
use_channel = 0
output_poc_image = False
poc_filename = None
poc_suffix = '_poc.tif'
fitting_size = 9

parser = argparse.ArgumentParser(description='Calculate sample drift using A-KAZE feature matching', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--output-file', default = output_filename, \
                    help='output TSV file name ({0} if not specified)'.format(output_filename))

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

parser.add_argument('-P', '--output-poc', action = 'store_true', \
                    help='output poc images')

parser.add_argument('-p', '--poc-file', default = poc_filename, \
                    help='filename to output poc images ([basename]{0} if not specified)'.format(poc_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
output_filename = args.output_file
ref_filename = args.ref_image
use_channel = args.use_channel
output_poc_image = args.output_poc
poc_filename = args.poc_file

if args.poc_file is None:
    poc_filename = mmtiff.with_suffix(input_filename, poc_suffix)
else:
    poc_filename = args.poc_file

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')

input_images = input_tiff.as_list(channel = use_channel, drop = True)

# hanning window
hanning_z = np.hanning(input_tiff.total_zstack)
hanning_y = np.hanning(input_tiff.height)
hanning_x = np.hanning(input_tiff.width)
mesh_z, mesh_y, mesh_x = np.meshgrid(hanning_z, hanning_y, hanning_x, indexing = 'ij')
hanning_mat = mesh_z * mesh_y * mesh_x

# read reference image
if ref_filename is None:
    ref_image = input_images[0].copy()
else:
    ref_tiff = mmtiff.MMTiff(ref_filename)
    if ref_tiff.colored:
        raise Exception('Reference image: color reference image not accepted.')
    ref_image = ref_tiff.as_list(use_channel = 0, drop = True)[0]

# calculate POC images (center = no drifting)
ref_fft_conj = np.conj(np.fft.fftn(ref_image * hanning_mat))

poc_image_list = []
for index in range(len(input_images)):
    image_fft = np.fft.fftn(input_images[index] * hanning_mat)
    corr_image = ref_fft_conj * image_fft / np.abs(ref_fft_conj * image_fft)
    poc_image = np.fft.fftshift(np.real(np.fft.ifftn(corr_image)))
    poc_image_list.append(poc_image)

if output_poc_image:
    print("Output POC images:", poc_filename)
    poc_image = np.array(poc_image_list)
    poc_image = poc_image[:, :, np.newaxis]
    input_tiff.save_image_ome(poc_filename, poc_image)

shift_list = []
shape = np.array(ref_image.shape)
center = shape // 2
for index in range(len(poc_image_list)):
    max_pos = ndimage.maximum_position(poc_image_list[index])
    max_val = poc_image_list[index][max_pos]
    shift = (max_pos - center).astype(float)
#
#    fitting_half = int(fitting_size // 2)
#    fitting_min = [max(0, max_pos[i] - fitting_half) for i in range(len(max_pos))]
#    fitting_max = [min(max_pos[i] + fitting_half + 1, shape[i]) for i in range(len(max_pos))]
#    slices = tuple([slice(fitting_min[i], fitting_max[i]) for i in range(len(max_pos))])
    #ranges = [range(fitting_min[i] - max_pos[i], fitting_max[i] - max_pos[i]) for i in range(len(max_pos))]
#    ranges = [range(fitting_min[i] - center[i], fitting_max[i] - center[i]) for i in range(len(max_pos))]

#    if shape[0] == 1:
#        meshes = np.meshgrid(*ranges[1:], indexing = 'ij')
#        values = poc_image_list[index][0][slices[1:]]
#        init_param = [max_val, 0.0, 0.0]
#        def sinc_fit (alpha, delta):
#            return alpha / (shape[0] * shape[1]) * \
#                np.sin(np.pi * (meshes[0] + delta[0]) / 2) / np.sin(np.pi * (meshes[0] + delta[0]) / shape[0]) * \
#                np.sin(np.pi * (meshes[1] + delta[1]) / 2) / np.sin(np.pi * (meshes[1] + delta[1]) / shape[1])
#        def error_func (params):
#            return np.ravel(sinc_fit(params[0], params[1:]) - values)
#        #print(sinc_fit(max_val, [0.0, 0.0]))
#        lsq_result = optimize.leastsq(error_func, init_param)
#        corr = lsq_result[0][0]
#        shift[1:] = shift[1:] + lsq_result[0][1:]
#    else:
#        init_param = [max_val, 0.0, 0.0, 0.0]
#        meshes = np.meshgrid(*ranges, indexing = 'ij')
#        values = poc_image_list[index][slices]
#        def sinc_fit (alpha, deltas):
#            return alpha / (shape[0] * shape[1] * shape[2]) * \
#                np.sin(np.pi * (meshes[0] + deltas[0])) / np.sin(np.pi * (meshes[0] + deltas[0]) / shape[0]) * \
#                np.sin(np.pi * (meshes[1] + deltas[1])) / np.sin(np.pi * (meshes[1] + deltas[1]) / shape[1]) * \
#                np.sin(np.pi * (meshes[2] + deltas[2])) / np.sin(np.pi * (meshes[1] + deltas[2]) / shape[2])
#        def error_func (params):
#            return np.ravel(sinc_fit(params[0], params[1:]) - values)
#        lsq_result = optimize.leastsq(error_func, init_param)
#        corr = lsq_result[0][0]
#        shift = shift + lsq_result[0][1:]

    print("Index:", index, "Shift:", shift, "Corr:", max_val)
    shift_list.append({'index': index, \
        'shift_x': shift[0], 'shift_y': shift[1], 'shift_z': shift[2], 'corr': max_val})

# open tsv file and write header
print("Output TSV:", output_filename)
shift_table = pd.DataFrame(shift_list)
shift_table.to_csv(output_filename, sep = '\t', index = False)
