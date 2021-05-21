#!/usr/bin/env python

import sys, argparse, numpy, math
from scipy.ndimage import rotate, zoom
from mmtools import mmtiff

# defaults
input_filename = None
filename_suffix = '_rotated.tif'
output_filename = None
gpu_id = None
rotation_range = [0, 90, 2]
rotation_axis = 'z'
expansion_factor = math.sqrt(2)
#expansion_factor = 2

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

parser.add_argument('-e', '--expansion_factor', default = expansion_factor, type = float, \
                    help='Expansion factor of the image size')

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
save_memory = args.save_memory
expansion_factor = args.expansion_factor
rotation_axis = args.rotation_axis.lower()

# expand to the end of range
rotation_range = args.rotation_range
rotation_range[1] = rotation_range[1] + rotation_range[2]

if args.output_file is None:
    output_filename = mmtiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read input image(s) and expand it
input_tiff = mmtiff.MMTiff(input_filename)
#input_image = input_tiff.as_array(channel = use_channel, drop = True)
input_image_list = input_tiff.as_list()

# load cupy if gpu_id is specified
if gpu_id is not None:
    import cupy
    from cupyx.scipy.ndimage import rotate as cupyx_rotate
    from cupyx.scipy.ndimage import zoom as cupyx_zoom

    device = cupy.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)

# set size and rotation tuple
resized_shape = numpy.array(input_image_list[0].shape).astype(numpy.float)
resized_shape = resized_shape[[0, 2, 3]]

# scale z
z_scale_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
resized_shape[0] = resized_shape[0] * z_scale_ratio

# expand the image area
print("Rotation axis:", rotation_axis)
if rotation_axis == 'z':
    resized_shape[1] = resized_shape[1] * expansion_factor
    resized_shape[2] = resized_shape[2] * expansion_factor
    axis_tuple = (1, 2)
elif rotation_axis == 'y':
    resized_shape[0] = resized_shape[0] * expansion_factor
    resized_shape[2] = resized_shape[2] * expansion_factor
    axis_tuple = (0, 2)
elif rotation_axis == 'x':
    resized_shape[0] = resized_shape[0] * expansion_factor
    resized_shape[1] = resized_shape[1] * expansion_factor
    axis_tuple = (0, 1)
else:
    raise Exception('Unknown axis:', rotation_axis)

resized_shape = resized_shape.astype(numpy.int)
print("Resized shape:", resized_shape)
output_image_list_all = []

for channel in range(input_tiff.total_channel):
    output_image_list_channels = []
    for index in range(input_tiff.total_time):
        print("Time:", index, "of", input_tiff.total_time)
        input_image = input_image_list[index][:, channel]
        zoom_factors = [z_scale_ratio, 1, 1]

        if gpu_id is None:
            zoomed_image = zoom(input_image, zoom = zoom_factors)
            resized_image = mmtiff.resize(zoomed_image, resized_shape, center = True)
        else:
            device = cupy.cuda.Device(gpu_id)
            mempool = cupy.get_default_memory_pool()
            if save_memory:
                if input_image.dtype.alignment > 4:
                    if input_image.dtype.kind == 'i':
                        data_type = numpy.int32
                    elif input_image.dtype.kind == 'u':
                        data_type = numpy.uint32
                    elif input_image.dtype.kind == 'f':
                        data_type = numpy.float32                
                    else:
                        raise Exception("Unsupported data type:", input_image.dtype)
                else:
                    data_type = input_image.dtype
            else:
                print("Memory saving disabled. The GPU may give an out-of-memory error.")
                data_type = input_image.dtype
            
            print("Using data type:", data_type.name)
            gpu_input_image = cupy.array(input_image, dtype = data_type)
            print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
            print("Free memory:", device.mem_info)

            gpu_zoomed_image = cupyx_zoom(gpu_input_image, zoom = zoom_factors, order = 1)
            print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
            print("Free memory:", device.mem_info)

            gpu_resized_image = cupy.zeros(resized_shape, dtype = data_type)
            slices_source, slices_target = \
                mmtiff.paste_slices(gpu_zoomed_image.shape, gpu_resized_image.shape, \
                                    center = True)
            gpu_resized_image[slices_target] = gpu_zoomed_image[slices_source].copy()
            print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
            print("Free memory:", device.mem_info)

            gpu_input_image = None
            gpu_zoomed_image = None
            print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
            print("Free memory:", device.mem_info)

        # prepare an empty array
        output_image_list = []
        print("Rotation:", end = ' ')
        for angle in range(*rotation_range):
            print(angle, end = ' ', flush = True)
            if gpu_id is None:
                rotated_image = rotate(resized_image, angle, axes = axis_tuple, \
                                    reshape = False)
                output_image_list.append(rotated_image)
            else:
                gpu_rotated_image = cupyx_rotate(gpu_resized_image, angle, \
                                                axes = axis_tuple, \
                                                order = 1, reshape = False)
                rotated_image = cupy.asnumpy(gpu_rotated_image)
                output_image_list.append(rotated_image)
        print(".")
        output_image_list_channels.extend(output_image_list)

    output_image_list_all.append(output_image_list_channels)

# output multipage tiff, dimensions should be in TZCYX order
print("Output image file to %s." % (output_filename))
output_array = numpy.array(output_image_list_all)
print(output_array.shape)
output_array = output_array.transpose((0, 1))
print(output_array.shape)

output_array = output_array.astype(input_tiff.dtype)
input_tiff.save_image_ome(output_filename, output_array)
