#!/usr/bin/env python

import sys, argparse, tifffile
import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from mmtools import mmtiff

# defaults
input_filename = None
output_filename = 'shift.txt'
ref_filename = None
use_channel = 0
output_shifted_image = False
shifted_image_filename = None
shifted_image_suffix = '_shifted.tif'
gpu_id = None

parser = argparse.ArgumentParser(description='Calculate sample shift using affine matrix and powell optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--output-file', default = output_filename, \
                    help='output TSV file name ({0} if not specified)'.format(output_filename))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

parser.add_argument('-S', '--output-shifted-image', action = 'store_true', \
                    help='output shifted images')

parser.add_argument('-s', '--shifted-image-file', default = shifted_image_filename, \
                    help='filename to output shifted images ([basename]{0} if not specified)'.format(shifted_image_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
output_filename = args.output_file
ref_filename = args.ref_image
use_channel = args.use_channel
gpu_id = args.gpu_id
output_shifted_image = args.output_shifted_image

if args.shifted_image_file is None:
    shifted_image_filename = mmtiff.with_suffix(input_filename, shifted_image_suffix)
else:
    shifted_image_filename = args.shifted_image_file

# activate GPU
if gpu_id is not None:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)

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

# hanning window
hanning_z = np.hanning(input_tiff.total_zstack)
hanning_y = np.hanning(input_tiff.height)
hanning_x = np.hanning(input_tiff.width)
mesh_z, mesh_y, mesh_x = np.meshgrid(hanning_z, hanning_y, hanning_x, indexing = 'ij')
hanning_mat = mesh_z * mesh_y * mesh_x

# prepare fft_conj of reference images (center = no drifting)
if gpu_id is None:
    ref_fft_conj = np.conj(np.fft.fftn(ref_image * hanning_mat))
else:
    hanning_mat = cp.array(hanning_mat)
    ref_fft_conj = cp.conj(cp.fft.fftn(cp.array(ref_image) * hanning_mat))

# calculate POCs for pre-registration
poc_image_list = []
shift_list = []
center = np.array(ref_image.shape) // 2
for index in range(len(input_images)):
    if gpu_id is None:
        image = input_images[index]
        image_fft = np.fft.fftn(image * hanning_mat)
        corr_image = ref_fft_conj * image_fft / np.abs(ref_fft_conj * image_fft)
        poc_image = np.fft.fftshift(np.real(np.fft.ifftn(corr_image)))
    else:
        image = cp.array(input_images[index])
        image_fft = cp.fft.fftn(image * hanning_mat)
        corr_image = ref_fft_conj * image_fft / cp.abs(ref_fft_conj * image_fft)
        poc_image = cp.fft.fftshift(cp.real(cp.fft.ifftn(corr_image)))
        poc_image = cp.asnumpy(poc_image)

    max_pos = ndimage.maximum_position(poc_image)
    max_val = poc_image[max_pos]
    image_shift = (max_pos - center).astype(float)
    print("POC performed. Index:", index, "Shift:", image_shift, "Corr:", max_val)
    shift_list.append({'index': index, \
        'init_x': image_shift[2], 'init_y': image_shift[1], 'init_z': image_shift[0], \
        'poc_corr': max_val})

# prepare normalized reference image
def normalize (image):
    clip_max = np.percentile(image, 99)
    clip_min = np.percentile(image, 1)
    return (image.clip(clip_min, clip_max) - clip_min) / (clip_max - clip_min)


if gpu_id is None:
    ref_float = normalize(ref_image)
    ref_sigma = np.std(ref_float)
else:
    ref_float = cp.array(normalize(ref_image))
    ref_sigma = cp.std(ref_float)

matrix_list = []
shifted_image_list = []
# optimization using an affine matrix
# note: input = matrix * output + offset
for index in range(len(input_images)):
    print("Starting optimization:", index)
    init_x = shift_list[index]['init_x']
    init_y = shift_list[index]['init_y']
    init_z = shift_list[index]['init_z']
    if input_tiff.total_zstack == 1:
        input_image = input_images[index][0]
        init_params = np.array([init_x, init_y])
        params_to_matrix = lambda p: np.array([[1.0, 0.0, p[1]], [0.0, 1.0, p[0]], [0.0, 0.0, 1.0]])
    else:
        input_image = input_images[index]
        init_params = np.array([init_x, init_y, init_z])
        params_to_matrix = lambda p: np.array([[1.0, 0.0, 0.0, p[2]],
                                               [0.0, 1.0, 0.0, p[1]],
                                               [0.0, 0.0, 1.0, p[0]],
                                               [0.0, 0.0, 0.0, 1.0]])

    if gpu_id is None:
        #image_float = input_image * hanning_mat
        image_float = normalize(input_image)
        def error_func (params):
            matrix = params_to_matrix(params)
            trans_float = ndimage.affine_transform(image_float, matrix)
            trans_sigma = np.std(trans_float)
            #error = np.sum(ref_float * trans_float * hanning_mat) / ref_sigma / trans_sigma
            error = np.sum(ref_float * trans_float * hanning_mat)
            #print("Trial:", -error)
            #print(matrix)
            return -error
    else:
        image_float = cp.array(normalize(input_image))
        #image_float = cp.array(input_image) * hanning_mat
        def error_func (params):
            matrix = cp.array(params_to_matrix(params))
            trans_float = cpimage.affine_transform(image_float, matrix)
            trans_sigma = cp.std(trans_float)
            #error = cp.asnumpy(cp.sum(ref_float * trans_float * hanning_mat) / ref_sigma / trans_sigma)
            error = cp.asnumpy(cp.sum(ref_float * trans_float * hanning_mat))
            #print("Trial:", -error)
            #print(matrix)
            return -error

    results = optimize.minimize(error_func, init_params, method = "Nelder-Mead")
    final_matrix = params_to_matrix(results.x)

    if output_shifted_image:
        if input_tiff.total_zstack == 1:
            input_image = input_images[index][0]
        else:
            input_image = input_images[index]

        if gpu_id is None:
            shifted_image = ndimage.affine_transform(input_image, final_matrix)
        else:
            shifted_image = cpimage.affine_transform(cp.array(input_image), cp.array(final_matrix))
            shifted_image = cp.asnumpy(shifted_image)

        if input_tiff.total_zstack == 1:
            shifted_image = shifted_image[np.newaxis, np.newaxis]
        else:
            shifted_image = shifted_image[:, np.newaxis]

        shifted_image_list.append(shifted_image)

    shift_list[index]['matrix'] = ','.join([str(x) for x in results.x])
    print("Optimization performed. Matrix:", shift_list[index]['matrix'])
    print(results)
    print(final_matrix)

# open tsv file and write header
print("Output TSV:", output_filename)
shift_table = pd.DataFrame(shift_list)
shift_table.to_csv(output_filename, sep = '\t', index = False)

# output shifted images
if output_shifted_image:
    print("Output shifted image:", shifted_image_filename)
    input_tiff.save_image_ome(shifted_image_filename, np.array(shifted_image_list))
