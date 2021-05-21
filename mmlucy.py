#!/usr/bin/env python

import sys, argparse, pathlib, numpy, tifffile, time
from scipy.ndimage import zoom
from mmtools import mmtiff, lucy

# defaults
psf_folder = pathlib.Path(__file__).parent.joinpath('psf')
input_filename = None
output_filename = None
output_suffix = '_dec.tif'
time_range = [0, 0]
psf_filename = 'diSPIM.tif'
iterations = 10
gpu_id = None

parser = argparse.ArgumentParser(description = 'Deconvolve images using the Richardson-Lucy algorhythm', \
                                 epilog = 'Psychiatric help: 5 cents. The doctor is *IN*.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-n', '--number-of-iterations', default = iterations, \
                    help='number of iterations')

parser.add_argument('-d', '--disable-z-scale-image', action = 'store_true', \
                    help='Disable scaling of image z dimension')

parser.add_argument('-f', '--z-scale-psf', action = 'store_true', \
                    help='Scale z dimension of psf')

parser.add_argument('-k', '--keep-z-scale', action = 'store_true', \
                    help='Keep z scaling of images after deconvolution')

parser.add_argument('-t', '--time-range', nargs = 2, type = int, default = time_range, \
                    metavar=('START', 'COUNT'), \
                    help='range of time to apply deconvolution (COUNT = 0 for all)')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-m', '--use-matrix', action = 'store_true', \
                    help='Disable FFT and use Matrix calculation.')

parser.add_argument('-s', '--save-memory', action = 'store_true', \
                    help='Save memory using float32 and complex64 (mainly for GPU)')

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to deconvolve')

args = parser.parse_args()

# defaults
time_range = args.time_range
iterations = args.number_of_iterations
psf_filename = args.psf_image
z_scale_image = not args.disable_z_scale_image
keep_z_scale = args.keep_z_scale
z_scale_psf = args.z_scale_psf
gpu_id = args.gpu_id
use_matrix = args.use_matrix
save_memory = args.save_memory

input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.stem(input_filename) + output_suffix
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

# setting image scale
if z_scale_image and input_tiff.total_zstack > 1:
    z_scale_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
    print("Setting z scaling of images:", z_scale_ratio)
else:
    z_scale_ratio = 1

# z-zoom psf
if z_scale_psf and input_tiff.total_zstack > 1:
    psf_z_scale_ratio = input_tiff.pixelsize_um / input_tiff.z_step_um
    psf_image = zoom(psf_image, (psf_z_scale_ratio, 1.0, 1.0))
    print("Scaling psf image into:", psf_image.shape)

# deconvolve
deconvolver = lucy.Lucy(psf_image, gpu_id, use_matrix, save_memory)

time_start = time_range[0]
if time_range[1] == 0:
    time_count = len(input_list)
else:
    time_count = min(len(input_list), time_range[1])

# save results in the CTZYX order
output_list = []
print("Start deconvolution:", time.ctime())
for channel in range(input_tiff.total_channel):
    print("Prosessing channel {0}:".format(channel))
    output_frames = []
    print("Frames:", end = ' ')
    for index in range(time_start, time_count):
        input_image = input_list[index][:, channel]
        print(index, end = ' ', flush = True)
        output_frames.append(deconvolver.deconvolve(input_image, iterations, z_scale_ratio, keep_z_scale))
    output_list.append(output_frames)
    print(".")
print("End deconvolution:", time.ctime())

# shape output into the TZCYX order
output_image = numpy.array(output_list)
output_axis = numpy.arange(output_image.ndim)
output_axis[0] = 1
output_axis[1] = 2
output_axis[2] = 0
output_image = output_image.transpose(output_axis)

if (input_tiff.dtype.kind == 'i' or input_tiff.dtype.kind == 'u') and \
        numpy.max(output_image) <= numpy.iinfo(input_tiff.dtype).max:
    output_image = mmtiff.float_to_int(output_image, input_tiff.dtype)
else:
    output_image = output_image.astype(numpy.float32)

# output in the ImageJ format, dimensions should be in TZCYX order
input_tiff.save_image(output_filename, output_image)
