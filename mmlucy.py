#!/usr/bin/env python

import sys, argparse, pathlib, numpy, tifffile
from mmtools import mmtiff, lucy

# defaults
psf_folder = pathlib.Path(__file__).parent.joinpath('psf')
input_filename = None
output_filename = None
output_suffix = '_dec.tif'
time_range = [0, 0]
psf_filename = 'diSPIM.tif'
iterations = 10
z_zoom_image = False
z_shrink_image = False
gpu_id = None
use_fft = False

parser = argparse.ArgumentParser(description='Deconvolve images using the Richardson-Lucy algorhythm', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-f', '--use-fft', action = 'store_true', default = use_fft, \
                    help='Use FFT in the CPU mode')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-n', '--number-of-iterations', default = iterations, \
                    help='number of iterations')

parser.add_argument('-z', '--z-zoom-image', action = 'store_true', default = z_zoom_image, \
                    help='Zoom z dimension to achieve xy_scale = z_scale')

parser.add_argument('-s', '--z-shrink-image', action = 'store_true', default = z_shrink_image, \
                    help='Shrink z dimension after deconvolution')

parser.add_argument('-t', '--time-range', nargs = 2, type = int, default = time_range, \
                    metavar=('START', 'COUNT'), \
                    help='range of time to apply deconvolution (COUNT = 0 for all)')

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to deconvolve')

args = parser.parse_args()

# defaults
time_range = args.time_range
iterations = args.number_of_iterations
psf_filename = args.psf_image
z_zoom_image = args.z_zoom_image
z_shrink_image = args.z_shrink_image
gpu_id = args.gpu_id
use_fft = args.use_fft

input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + output_suffix
    if output_filename == input_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

# load psf image
if pathlib.Path(psf_filename).exists():
    print("Read PSF image in the current folder:", psf_filename)
    psf_image = tifffile.imread(psf_filename)
else:
    psf_path = pathlib.Path(psf_folder).joinpath(psf_filename)
    if psf_path.exists():
        print("Read PSF image in the system folder:", str(psf_path))
        psf_image = tifffile.imread(str(psf_path))
    else:
        raise Exception('PSF file {0} not found'.format(psf_filename))

# load input image
input_tiff = mmtiff.MMTiff(input_filename)
input_list = input_tiff.as_list()

# deconvolve
deconvolver = lucy.Lucy(psf_image, gpu_id, use_fft)

time_start = time_range[0]
if time_range[1] == 0:
    time_count = len(input_list)
else:
    time_count = min(len(input_list), time_range[1])

output_list = []
for index in range(time_start, time_count):
    output_channels = []
    for channel in range(input_tiff.total_channel):
        input_image = input_list[index][:, channel]
        if z_zoom_image:
            zoom_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
        else:
            zoom_ratio = 1
        print("Processing time frame: {0}, channel: {1}, z-zoom: {2}".format(index, channel, zoom_ratio))
        output_channels.append(deconvolver.deconvolve(input_image, iterations, zoom_ratio, z_shrink_image))
    output_list.append(output_channels)

# shape output
output_image = numpy.array(output_list)
output_axis = numpy.arange(output_image.ndim)
output_axis[1] = 2
output_axis[2] = 1
output_image = output_image.transpose(output_axis)

if input_tiff.dtype.kind == 'i' or input_tiff.dtype.kind == 'u':
    print("Converting from float to int:", input_tiff.dtype.name)
    output_image = mmtiff.MMTiff.float_to_int(output_image, input_tiff.dtype)

# output in the ImageJ format, dimensions should be in TZCYX order
input_tiff.save_image(output_filename, output_image)