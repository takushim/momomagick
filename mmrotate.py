#!/usr/bin/env python

import sys, argparse, numpy, math
from scipy.ndimage import rotate
from mmtools import mmtiff

# defaults
input_filename = None
filename_suffix = '_rotated.tif'
output_filename = None
gpu_id = None
rotation_range = [0, 90, 2]
rotation_axis = 'z'

parser = argparse.ArgumentParser(description='Rotate a multipage TIFF image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-s', '--save-memory', action = 'store_true', \
                    help='Save memory using float32 and complex64 (only for GPU)')

parser.add_argument('-r', '--rotation-range', nargs=3, type=float, \
                    default = rotation_range, \
                    metavar=('BEGIN', 'END', 'STEP'), \
                    help='Range of rotation angles')

parser.add_argument('-a', '--rotation-axis', default = rotation_axis, \
                    choices = ['x', 'y', 'z', 'X', 'Y', 'Z'], \
                    help='Axis to rotate the image')

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
save_memory = args.save_memory
rotation_axis = args.rotation_axis.lower()

# expand to the end of range
rotation_range = args.rotation_range
rotation_range[1] = rotation_range[1] + rotation_range[2]

if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# load cupy if gpu_id is specified
if gpu_id is not None:
    import cupy
    from cupyx.scipy.ndimage import rotate as cupyx_rotate

    device = cupy.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))

# read input image(s) and expand it
input_tiff = mmtiff.MMTiff(input_filename)
#input_image = input_tiff.as_array(channel = use_channel, drop = True)
input_image_list = input_tiff.as_list()

# set size and rotation tuple
resized_shape = numpy.array(input_image_list[0].shape).astype(numpy.float)
if rotation_axis == 'z':
    diagonal = math.sqrt(resized_shape[2]**2 + resized_shape[3]**2)
    resized_shape[2] = diagonal
    resized_shape[3] = diagonal
    resized_shape = resized_shape.astype(numpy.int)
    axis_tuple = (2, 3)
elif rotation_axis == 'y':
    diagonal = math.sqrt(resized_shape[0]**2 + resized_shape[3]**2)
    resized_shape[0] = diagonal
    resized_shape[3] = diagonal
    resized_shape = resized_shape.astype(numpy.int)
    axis_tuple = (0, 3)
elif rotation_axis == 'x':
    diagonal = math.sqrt(resized_shape[0]**2 + resized_shape[2]**2)
    resized_shape[0] = diagonal
    resized_shape[2] = diagonal
    resized_shape = resized_shape.astype(numpy.int)
    axis_tuple = (0, 2)
else:
    raise Exception('Unknown axis:', rotation_axis)

print("Resized shape:", resized_shape)
output_image_list_all = []

for index in range(input_tiff.total_time):
    print("Time:", index, "of", input_tiff.total_time)
    input_image = input_image_list[index]
    resized_image = mmtiff.MMTiff.resize(input_image, resized_shape, center = True)

    if gpu_id is not None:
        if save_memory:
            if resized_image.dtype.kind == 'i':
                gpu_resized_image = cupy.array(resized_image, dtype = numpy.int32)
            elif resized_image.dtype.kind == 'u':
                gpu_resized_image = cupy.array(resized_image, dtype = numpy.uint32)
            elif resized_image.dtype.kind == 'f':
                gpu_resized_image = cupy.array(resized_image, dtype = numpy.float32)
            else:
                raise Exception("Unsupported data type:", resized_image.dtype)
        else:
            print("Memory saving disabled. The GPU may give an out-of-memory error.")
            gpu_resized_image = cupy.array(resized_image)

    # prepare an empty array
    output_image_list = []
    print("Rotation:", end = ' ')
    for angle in range(*rotation_range):
        print(angle, end = ' ', flush = True)
        if gpu_id is None:
            rotated_image = rotate(resized_image, angle, axes = axis_tuple, \
                                   reshape = False)
            print(rotated_image.shape)
            output_image_list.append(rotated_image)
        else:
            gpu_rotated_image = cupyx_rotate(gpu_resized_image, angle, \
                                            axes = axis_tuple, \
                                            order = 1, reshape = False)
            output_image_list.append(cupy.asnumpy(gpu_rotated_image))
    print(".")

    output_image_list_all.extend(output_image_list)

# output multipage tiff, dimensions should be in TZCYX order
print("Output image file to %s." % (output_filename))
output_array = numpy.array(output_image_list_all)
print(output_array.shape)

output_array = output_array.astype(input_tiff.dtype)
input_tiff.save_image_ome(output_filename, output_array)
