#!/usr/bin/env python

import tifffile
import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path
from logging import getLogger
from tqdm import tqdm
from ome_types import to_xml, from_xml, OME
from ome_types.model import Image, Pixels, TiffData, Channel
from ome_types.model.simple_types import PixelType, ChannelID, UnitsLength, UnitsTime, Color
from mmtools import log
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
        logger.warning("GPU ID not specified. Continuing with CPU.")
        return None

    device = cp.cuda.Device(gpu_id)
    device.use()
    logger.info("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    logger.info("Free memory:", device.mem_info)
    return device

def with_suffix (filename, suffix):
    filename = Path(filename).with_suffix('')
    if filename.suffix.lower() == '.ome':
        filename = filename.with_suffix('')
    return str(filename) + suffix

dtype_to_ometype = {
    np.dtype(np.int8): PixelType.INT8,
    np.dtype(np.int16): PixelType.INT16,
    np.dtype(np.int32): PixelType.INT32,
    np.dtype(np.uint8): PixelType.UINT8,
    np.dtype(np.uint16): PixelType.UINT16,
    np.dtype(np.uint32): PixelType.UINT32,
    np.dtype(np.float32): PixelType.FLOAT,
    np.dtype(np.float64): PixelType.DOUBLE,
    np.dtype(np.complex64): PixelType.COMPLEXFLOAT,
    np.dtype(np.complex128): PixelType.COMPLEXDOUBLE,
}

ome_colors = [Color(0xFF000000), Color(0x00FF0000), Color(0x0000FF00), Color(0x000000FF)]

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
                metadata = self.__read_metadata(tiff, series = series)

            if 'T' not in axes:
                image_array = image_array[np.newaxis]
            if 'Z' not in axes:
                image_array = image_array[:, np.newaxis]
            if 'C' not in axes:
                image_array = image_array[:, :, np.newaxis]

            image_array = image_array.swapaxes(1, 2)
            if ('S' in axes) and (keep_s_axis == False):
                image_array = self.__concat_s_channel(image_array)

            self.image_array = image_array
            self.update_dimensions()
            self.__set_metadata(metadata)

            logger.debug("Image shaped into: {0} {1}".format(str(self.image_array.shape), self.axes))

        except OSError:
            self.reset_stack()
            raise

    def __concat_s_channel (self, image_array):
        image_list = [image_array[..., index] for index in range(image_array.shape[-1])]
        image_array = np.concatenate(image_list, axis = 1)
        return image_array

    def save_imagej_tiff (self, filename):
        logger.debug("Saving ImageJ. Shape: {0}. Type: {1}".format(self.image_array.shape, self.image_array.dtype))
        resolution = (1 / self.pixel_um[2], 1 / self.pixel_um[1])
        z_step_um = self.pixel_um[0]
        metadata = {'spacing': z_step_um, 'unit': 'um', 'Composite mode': 'composite', 'finterval': self.finterval_sec}
        tifffile.imwrite(filename, self.image_array.swapaxes(1, 2), imagej = True, \
                         resolution = resolution, metadata = metadata)

    def save_ome_tiff (self, filename, bigtiff = False):
        logger.debug("Saving OME. Shape: {0}. Type: {1}".format(self.image_array.shape, self.image_array.dtype))
        if self.has_s_axis:
            image_array = self.__concat_s_channel(self.image_array)
            c_count = image_array.shape[1]
            samples_per_pixel = self.has_s_axis
        else:
            image_array = self.image_array
            c_count = self.c_count
            samples_per_pixel = 1

        ome_pixels = Pixels(id = "Pixels:0", dimension_order = 'XYZCT', \
                           type = dtype_to_ometype[self.image_array.dtype], \
                           size_t = self.t_count, size_c = c_count, \
                           size_z = self.z_count, size_y = self.height, size_x = self.width, \
                           interleaved = True if self.has_s_axis else None)

        ome_pixels.physical_size_x = self.pixel_um[2]
        ome_pixels.physical_size_y = self.pixel_um[1]
        ome_pixels.physical_size_z = self.pixel_um[0]
        ome_pixels.physical_size_x_unit = UnitsLength.MICROMETER
        ome_pixels.physical_size_y_unit = UnitsLength.MICROMETER
        ome_pixels.physical_size_z_unit = UnitsLength.MICROMETER
        ome_pixels.time_increment = self.finterval_sec
        ome_pixels.time_increment_unit = UnitsTime.SECOND

        ome_pixels.tiff_data_blocks = [TiffData(plane_count = self.t_count * c_count * self.z_count, ifd = 0)]
        ome_pixels.channels = [Channel(samples_per_pixel = samples_per_pixel, \
                                       id = ChannelID("Channel:0:{0}".format(index))) \
                               for index in range(c_count)]

        if self.has_s_axis:
            for index in range(c_count):
                ome_pixels.channels[index].color = ome_colors[index % 4]

        ome_image = Image(name = filename, id = "Image:0", pixels = ome_pixels)
        ome_xml = to_xml(OME(images = [ome_image])).encode()

        with open(filename, "wb") as fileio:
            with tifffile.TiffWriter(fileio, bigtiff = bigtiff) as tiff:
                tiff.write(image_array, description = ome_xml, metadata = None)

    def update_dimensions (self):
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

    def __read_metadata (self, tiff, series = 0):
        metadata = {}

        if 'XResolution' in tiff.pages[series].tags:
            values = tiff.pages[series].tags['XResolution'].value
            metadata['x_pixel_um'] = float(values[1]) / float(values[0])
        else:
            metadata['x_pixel_um'] = default_pixel_um

        if 'YResolution' in tiff.pages[series].tags:
            values = tiff.pages[series].tags['YResolution'].value
            metadata['y_pixel_um'] = float(values[1]) / float(values[0])
        else:
            metadata['y_pixel_um'] = default_pixel_um

        if tiff.imagej_metadata is not None:
            metadata['z_step_um'] = tiff.imagej_metadata.get('spacing', default_z_step_um)
            metadata['finterval_sec'] = tiff.imagej_metadata.get('finterval_sec', default_finterval_sec)
            logger.debug('Read imagej metadata: {0}'.format(str(metadata)))
        elif tiff.ome_metadata is not None:
            ome = from_xml(tiff.ome_metadata)
            if hasattr(ome.images[series].pixels, "physical_size_z"):
                metadata['z_step_um'] = ome.images[series].pixels.physical_size_z
            else:
                metadata['z_step_um'] = default_z_step_um

            if hasattr(ome.images[series].pixels, "time_increment"):
                metadata['finterval_sec'] = ome.images[series].pixels.time_increment
            else:
                metadata['finterval_sec'] = default_finterval_sec

            logger.debug('Read ome metadata: {0}'.format(str(metadata)))
        else:
            metadata['z_step_um'] = default_z_step_um
            metadata['finterval_sec'] = default_finterval_sec

        return metadata

    def __set_metadata (self, metadata):
        self.pixel_um = [metadata['z_step_um'], metadata['y_pixel_um'], metadata['x_pixel_um']]
        self.finterval_sec = metadata['finterval_sec']
    
    def __pasting_slices (self, src_shape, tgt_shape, centering = False, offset = None):
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
        t_slice = slice(0, self.t_count, 1)
        c_slice = slice(0, self.c_count, 1)

        new_array = self.__new_array(shape)
        current_shape = [self.z_count, self.height, self.width]
        slices_src, slices_tgt = self.__pasting_slices(current_shape, shape, centering = centering)
        new_array[tuple([t_slice, c_slice] + slices_tgt)] = self.image_array[tuple([t_slice, c_slice] + slices_src)]

        self.image_array = new_array
        self.update_dimensions()

    def crop (self, slice_list):
        self.image_array = self.image_array[tuple(slice_list)].copy()
        self.update_dimensions()

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
            yield t_index

        self.image_array = np.array(output_frames)
        self.update_dimensions()

    def apply_all (self, image_func, progress = False):
        if progress:
            with tqdm(total = self.t_count - 1) as pbar:
                for index in self.__apply_all(image_func):
                    pbar.n = index
                    pbar.refresh()
        else:
            self.__apply_all(image_func)

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

            self.apply_all(zoom_func)
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

        self.apply_all(rotate_func)

    def affine_transform (self, matrix, gpu_id = None):
        def affine_func (image):
            if gpu_id is None:
                image = ndimage.affine_transform(image, matrix, mode = 'grid-constant')
            else:
                image = cpimage.affine_transform(cp.array(image), cp.array(matrix), mode = 'grid-constant')
                image = cp.asnumpy(image)
            return image

        self.apply_all(affine_func)

    def shift (self, offset, gpu_id = None):
        def shift_func (image):
            if gpu_id is None:
                image = ndimage.interpolation.shift(image, offset)
            else:
                image = cp.asnumpy(cpimage.interpolation.shift(cp.array(image), offset))
            return image

        self.apply_all(shift_func)
