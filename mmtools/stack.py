#!/usr/bin/env python

import tifffile
import numpy as np
import scipy.ndimage as ndimage
from logging import getLogger
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

logger = getLogger(__name__)

default_pixel_um = 0.1625
default_z_step_um = 0.5
default_finterval_sec = 1

def turn_on_gpu (gpu_id):
    if gpu_id is None:
        print("GPU ID not specified. Continuing with CPU.")
        return None

    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)
    return device

class Stack:
    def __init__ (self, fileio = None, series = 0, keep_s_axis = False):
        if fileio is None:
            self.reset_stack()
        else:
            try:
                self.read_image(fileio, series = series, keep_s_axis = keep_s_axis)
            except OSError:
                self.reset_stack()
                raise

    def reset_stack (self):
        self.pixel_um = None
        self.finterval_sec = None
        self.z_count = None
        self.t_count = None
        self.c_count = None
        self.s_count = None
        self.height = None
        self.width = None
        self.axes = None
        self.has_s_axis = False
        self.image_array = None

    def read_image (self, fileio, series = 0, keep_s_axis = False):
        try:
            self.reset_stack()

            with tifffile.TiffFile(fileio) as tiff:
                axes = tiff.series[series].axes
                image_array = tiff.asarray(series = series)
                metadata = self.__read_metadata(tiff)

            if 'T' not in axes:
                image_array = image_array[np.newaxis]
            if 'Z' not in axes:
                image_array = image_array[:, np.newaxis]
            if 'C' not in axes:
                image_array = image_array[:, :, np.newaxis]

            image_array = image_array.swapaxes(1, 2)
            if ('S' in axes) and (keep_s_axis == False):
                image_list = [image_array[..., index] for index in range(image_array.shape[-1])]
                image_array = np.stack(image_list, axis = 1)

            self.image_array = image_array
            self.__update_dimensions()
            self.__set_metadata(metadata)

        except OSError:
            self.reset_stack()
            raise

    def save_imagej_tiff (self, filename):
        resolution = (1 / self.pixel_um[2], 1 / self.pixel_um[1])
        z_step_um = self.pixel_um[0]
        metadata = {'spacing': z_step_um, 'unit': 'um', 'Composite mode': 'composite', 'finterval': self.finterval_sec}
        tifffile.imwrite(filename, self.image_array.swapaxes(1, 2), imagej = True, \
                         resolution = resolution, metadata = metadata)

    def save_ome_tiff (self, filename):
        if self.has_s_axis:
            axes = 'TZCYXS'
        else:
            axes = 'TZCYX'

        resolution = (1 / self.pixel_um[2], 1 / self.pixel_um[1])
        metadata = {'PhysicalSizeX': self.pixel_um[2], 'PhysicalSizeXUnit': '\u03BCm',\
                    'PhysicalSizeY': self.pixel_um[1], 'PhysicalSizeYUnit': '\u03BCm',\
                    'PhysicalSizeZ': self.pixel_um[0], 'PhysicalSizeZUnit': '\u03BCm',\
                    'TimeIncrement': self.finterval_sec, 'TimeIncrementUnit': 's'}

        with tifffile.TiffWriter(filename, ome = True) as tiff:
            tiff.write(self.image_array.swapaxes(1, 2), metadata = metadata)

    def __update_dimensions (self):
        self.t_count = self.image_array.shape[0]
        self.c_count = self.image_array.shape[1]
        self.z_count = self.image_array.shape[2]
        self.height = self.image_array.shape[3]
        self.width = self.image_array.shape[4]
        if len(self.image_array.shape) > 5:
            self.s_count = self.image_array.shape[5]
            self.has_s_axis = True
            self.axes = 'TCZYXS'
        else:
            self.s_count = 0
            self.has_s_axis = False
            self.axes = 'TCZYX'

    def __read_metadata (self, tiff):
        metadata = {}

        if 'XResolution' in tiff.pages[0].tags:
            values = tiff.pages[0].tags['XResolution'].value
            metadata['x_pixel_um'] = float(values[1]) / float(values[0])
        else:
            metadata['x_pixel_um'] = default_pixel_um

        if 'YResolution' in tiff.pages[0].tags:
            values = tiff.pages[0].tags['YResolution'].value
            metadata['y_pixel_um'] = float(values[1]) / float(values[0])
        else:
            metadata['y_pixel_um'] = default_pixel_um

        if tiff.imagej_metadata is not None:
            logger.debug('Read imagej metadata: {0}'.format(str(metadata)))
            metadata['z_step_um'] = tiff.imagej_metadata.get('spacing', default_z_step_um)
            metadata['finterval_sec'] = tiff.imagej_metadata.get('finterval_sec', default_finterval_sec)
        elif tiff.ome_metadata is not None:
            logger.debug('Read ome metadata: {0}'.format(str(metadata)))
            metadata['z_step_um'] = tiff.ome_metadata.get('spacing', default_z_step_um)
            metadata['finterval_sec'] = tiff.ome_metadata.get('finterval_sec', default_finterval_sec)
        else:
            metadata['z_step_um'] = default_z_step_um
            metadata['finterval_sec'] = default_finterval_sec

        return metadata

    def __set_metadata (self, metadata):
        self.pixel_um = [metadata['z_step_um'], metadata['y_pixel_um'], metadata['x_pixel_um']]
        self.finterval_sec = metadata['finterval_sec']
    
    def __pasting_slices (src_shape, tgt_shape, centering = False, offset = None):
        if centering:
            shifts = (np.array(tgt_shape) - np.array(src_shape)) // 2
        else:
            shifts = np.array([0] * len(src_shape))

        if offset is not None:
            shifts = shifts + np.array(offset)

        src_starts = [min(-x, y) if x < 0 else 0 for x, y in zip(shifts, src_shape)]
        src_bounds = np.minimum(src_shape, tgt_shape - shifts)
        slices_src = tuple([slice(x, y) for x, y in zip(src_starts, src_bounds)])

        tgt_starts = [0 if x < 0 else min(x, y) for x, y in zip(shifts, tgt_shape)]
        tgt_bounds = np.minimum(src_shape + shifts, tgt_shape)
        slices_tgt = tuple([slice(x, y) for x, y in zip(tgt_starts, tgt_bounds)])

        return [slices_src, slices_tgt]

    def __new_array (self, shape):
        if self.has_s_axis:
            new_shape = [self.t_count, self.c_count, *shape, self.s_count]
        else:
            new_shape = [self.t_count, self.c_count, *shape]

        return np.zeros(new_shape, dtype = self.image_array.dtype)

    def resize (self, shape, centering = False):
        new_array = self.__new_array(shape)
        slices_src, slices_tgt = self.__pasting_slices(self.image_array.shape, shape, centering = centering)
        new_array[:, :, slices_tgt] = self.image_array[:, :, slices_src]
        self.image_array = new_array
        self.__update_dimensions()

    def crop (self, shape, offset):
        new_array = self.__new_array(shape)
        slices_src, slices_tgt = self.__pasting_slices(self.image_array.shape, shape, offset = -offset)
        new_array[:, :, slices_tgt] = self.image_array[:, :, slices_src]
        self.image_array = new_array
        self.__update_dimensions()

    def __apply_all (self, image_func):
        output_frames = []
        for t_index in range(self.t_count):
            output_channels = []
            for c_index in range(self.c_count):
                if self.has_s_axis:
                    output_images = []
                    for s_index in range(self.s_count):
                        image = self.image_array[t_index, c_index, ..., s_index]
                        image = image_func(image)
                        output_images.append(image)
                    output_channels.append(output_images)
                else:
                    image = self.image_array[t_index, c_index]
                    image = image_func(image)
                    output_channels.append(image)
            output_frames.append(output_channels)
        return np.array(output_frames)

    def scale_by_ratio (self, ratio = 1.0, gpu_id = None):
        if isinstance(ratio, (list, tuple, np.ndarray)):
            if len(ratio) == 0:
                ratio = [1.0, 1.0, 1.0]
            elif len(ratio) == 1:
                ratio = [ratio[0], ratio[0], ratio[0]]
            elif len(ratio) == 2:
                ratio = [ratio[0], ratio[1], ratio[1]]
        else:
            ratio = [ratio, ratio, ratio]

        if np.allclose(ratio, 1.0) == False:
            def zoom_func (image):
                if gpu_id is None:
                    image = ndimage.zoom(image, ratio)
                else:
                    image = cp.asnumpy(cpimage.zoom(image, ratio))
                return image

            self.image_array = self.__apply_all(zoom_func)
            self.__update_dimensions()
            self.pixel_um = [self.pixel_um[i] / ratio[i] for i in range(len(self.pixel_um))]

    def scale_by_pixelsize (self, pixel_um, gpu_id = None):
        if isinstance(pixel_um, (list, tuple, np.ndarray)):
            if len(pixel_um) == 0:
                pixel_um = self.pixel_um
            elif len(pixel_um) == 1:
                pixel_um = [pixel_um[0], pixel_um[0], pixel_um[0]]
            elif len(pixel_um) == 2:
                pixel_um = [pixel_um[0], pixel_um[1], pixel_um[1]]
        else:
            pixel_um = [pixel_um, pixel_um, pixel_um]

        ratio = [self.pixel_um[i] / pixel_um[i] for i in range(len(self.pixel_um))]
        self.scale_by_ratio(ratio = ratio, gpu_id = gpu_id)

    def rotate (self, angle = 0.0, axis = 0, gpu_id = None):
        if axis == 0 or axis == 'z' or axis == 'Z':
            rotate_tuple = (1, 2)
        elif axis == 1 or axis == 'y' or axis == 'Y':
            rotate_tuple = (0, 2)
        elif axis == 2 or axis == 'x' or axis == 'X':
            rotate_tuple = (0, 1)
        else:
            raise Exception('Invalid axis was specified.')

        def rotate_func (image):
            if gpu_id is None:
                image = ndimage.rotate(image, angle, axes = rotate_tuple, reshape = False)
            else:
                image = cpimage.rotate(cp.asarray(image), angle, axes = rotate_tuple, order = 1, reshape = False)
                image = cp.asnumpy(image)
            return image

        self.image_array = self.__apply_all(rotate_func)
        self.__update_dimensions()

    def affine_transform (self, matrix, gpu_id = None):
        def affine_func (image):
            if gpu_id is None:
                image = ndimage.affine_transform(image, matrix, mode = 'grid-constant')
            else:
                image = cpimage.affine_transform(cp.array(image), cp.array(matrix), mode = 'grid-constant')
                image = cp.asnumpy(image)
            return image

        self.image_array = self.__apply_all(affine_func)
        self.__update_dimensions()

    def shift (self, offset, gpu_id = None):
        def shift_func (image):
            if gpu_id is None:
                image = ndimage.interpolation.shift(image, offset)
            else:
                image = cp.asnumpy(cpimage.interpolation.shift(cp.array(image), offset))
            return image

        self.image_array = self.__apply_all(shift_func)
        self.__update_dimensions()
