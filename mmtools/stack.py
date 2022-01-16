#!/usr/bin/env python

import io, tifffile, json
import numpy as np
from pathlib import Path
from logging import getLogger
from progressbar import ProgressBar
from ome_types import to_xml, from_xml, OME
from ome_types.model import Image, Pixels, TiffData, Channel
from ome_types.model.simple_types import PixelType, ChannelID, UnitsLength, UnitsTime, Color
from . import gpuimage

logger = getLogger(__name__)

default_pixel_um = 0.1625
default_z_step_um = 0.5
default_finterval_sec = 1
default_voxel = [default_z_step_um, default_pixel_um, default_pixel_um]
default_shape = (1, 1, 1, 256, 256)
default_dtype = np.uint16

def turn_on_gpu (gpu_id):
    return gpuimage.turn_on_gpu(gpu_id)

def with_suffix (filename, suffix):
    filename = Path(filename).with_suffix('')
    if filename.suffix.lower() == '.ome':
        filename = filename.with_suffix('')
    return str(filename) + suffix

ome_size_limit = int(0.9 * (2 ** 31))

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

ome_ratio_to_um = {
    UnitsLength.METER: 1.0e-6,
    UnitsLength.MILLIMETER: 1.0e-3,
    UnitsLength.MICROMETER: 1.0,
    UnitsLength.NANOMETER: 1.0e3,
    UnitsLength.PICOMETER: 1.0e6,
    UnitsLength.ANGSTROM: 1.0e-1,
    UnitsLength.INCH: 25.4e4,
}

ome_ratio_to_sec = {
    UnitsTime.HOUR: 3600.0,
    UnitsTime.MINUTE: 60.0,
    UnitsTime.SECOND: 1.0,
    UnitsTime.MILLISECOND: 1.0e-3,
    UnitsTime.MICROSECOND: 1.0e-6,
    UnitsTime.NANOSECOND: 1.0e-9,
    UnitsTime.PICOSECOND: 1.0e-12,
}

ome_grayscale = Color(0xFFFFFF00)

ome_rgb_colors = [
    Color(0xFF000000), # Red
    Color(0x00FF0000), # Green
    Color(0x0000FF00), # Blue
    ] #

ome_multi_colors = [
    Color(0xFF000000), # Red
    Color(0x00FF0000), # Green
    Color(0x0000FF00), # Blue
    Color(0x00FFFF00), # Cyan
    Color(0xFF00FF00), # Magenta
    Color(0xFFFF0000), # Yellow
    Color(0xFFFFFF00), # Gray
    ]

imagej_ratio_to_um = {
    'm': 1.0e6,
    'mm': 1.0e3,
    'um': 1.0,
    '\u03BCm': 1.0,
    'nm': 1.0e-3,
    'pm': 1.0e-6,
}

resunit_ratio_to_um = {
    1: 1.0,     # no definition
    2: 25.4e3,  # inch
    3: 1.0e4,   # cm
}

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
        self.voxel_um = None
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

    def alloc_zero_image (self, shape = default_shape, dtype = default_dtype, \
                          voxel_um = default_voxel, finterval_sec = default_finterval_sec):
        self.image_array = np.zeros(shape, dtype = dtype)
        self.voxel_um = voxel_um
        self.finterval_sec = finterval_sec
        self.update_dimensions()

    def archive_properties (self):
        settings = {'voxel_um': self.voxel_um,
                    'finterval_sec': self.finterval_sec,
                    'z_count': self.z_count,
                    't_count': self.t_count,
                    'c_count': self.c_count,
                    'height': self.height,
                    'width': self.width,
                    'has_s_axis': self.has_x_axis,
                    's_count': self.s_count,
                    'colored': self.colored,
                    'axes': self.axes}
        return settings

    def read_image (self, fileio, series = 0, keep_s_axis = False):
        try:
            self.reset_stack()

            with tifffile.TiffFile(fileio) as tiff:
                axes = tiff.series[series].axes.upper()
                image_array = tiff.asarray(series = series)
                metadata = self.__read_metadata(tiff, series = series)

            if (set(axes) <= {'T', 'C', 'Z', 'Y', 'X', 'S'}) == False:
                raise Exception('Unknown axis format: {0}'.format(axes))

            for axis in 'ZCTYX':
                if axis not in axes:
                    image_array = image_array[np.newaxis]
                    axes = axis + axes

            if 'S' in axes:
                axis_order = [axes.find(axis) for axis in 'TCZYXS']
            else:
                axis_order = [axes.find(axis) for axis in 'TCZYX']

            logger.debug("Current axis: {0}. Order: {1}.".format(axes, axis_order))
            image_array = image_array.transpose(axis_order)

            if ('S' in axes) and (keep_s_axis == False):
                logger.info("The S axis is converted to the C axis.")
                image_array = self.__concat_s_channel(image_array)

            self.image_array = image_array
            self.update_dimensions()
            self.__set_metadata(metadata)

            logger.debug("Image shaped into: {0} {1}".format(str(self.image_array.shape), self.axes))

        except OSError:
            self.reset_stack()
            raise

    def read_image_by_chunk (self, fileio, series = 0, keep_s_axis = False, chunk_size = 1024 * 1024):
        try:
            byte_data = bytearray()
            with open(fileio, 'rb') as file:
                while len(chunk := file.read(chunk_size)) > 0:
                    byte_data.extend(chunk)
                    yield len(byte_data)

            with io.BytesIO(byte_data) as bytes_io:
                self.read_image(bytes_io, series = series, keep_s_axis = keep_s_axis)

        except OSError:
            raise

    def __concat_s_channel (self, image_array):
        image_list = [image_array[..., index] for index in range(image_array.shape[-1])]
        image_array = np.concatenate(image_list, axis = 1)
        return image_array

    def save_imagej_tiff (self, filename, dtype = None):
        logger.debug("Saving ImageJ. Shape: {0}. Type: {1}".format(self.image_array.shape, self.image_array.dtype))
        if dtype is None:
            output_array = self.image_array
        else:
            logger.info("Changing dtype: {0}".format(dtype))
            output_array = self.image_array.astype(dtype)

        resolution = (1 / self.voxel_um[2], 1 / self.voxel_um[1])
        z_step_um = self.voxel_um[0]
        metadata = {'spacing': z_step_um, 'unit': 'um', 'Composite mode': 'composite', 'finterval': self.finterval_sec}
        tifffile.imwrite(filename, output_array.swapaxes(1, 2), imagej = True, \
                         resolution = resolution, metadata = metadata)

    def save_ome_tiff (self, filename, dtype = None, bigtiff = None):
        logger.debug("Saving OME. Shape: {0}. Type: {1}".format(self.image_array.shape, self.image_array.dtype))
        if dtype is None:
            output_array = self.image_array
        else:
            logger.info("Changing dtype: {0}".format(dtype))
            output_array = self.image_array.astype(dtype)

        if bigtiff is None:
            if output_array.nbytes > ome_size_limit:
                logger.info("Saving in a BigTiff format. Size: {0}.".format(output_array.nbytes))
                bigtiff = True
            else:
                bigtiff = False

        if self.has_s_axis:
            output_array = self.__concat_s_channel(output_array)
            c_count = output_array.shape[1]
            samples_per_pixel = self.has_s_axis
        else:
            c_count = self.c_count
            samples_per_pixel = 1

        ome_pixels = Pixels(id = "Pixels:0", dimension_order = 'XYZCT', \
                           type = dtype_to_ometype[output_array.dtype], \
                           size_t = self.t_count, size_c = c_count, \
                           size_z = self.z_count, size_y = self.height, size_x = self.width, \
                           interleaved = True if self.has_s_axis else None)

        ome_pixels.physical_size_x = self.voxel_um[2]
        ome_pixels.physical_size_y = self.voxel_um[1]
        ome_pixels.physical_size_z = self.voxel_um[0]
        ome_pixels.physical_size_x_unit = UnitsLength.MICROMETER
        ome_pixels.physical_size_y_unit = UnitsLength.MICROMETER
        ome_pixels.physical_size_z_unit = UnitsLength.MICROMETER
        ome_pixels.time_increment = self.finterval_sec
        ome_pixels.time_increment_unit = UnitsTime.SECOND

        ome_pixels.tiff_data_blocks = [TiffData(plane_count = self.t_count * c_count * self.z_count, ifd = 0)]
        ome_pixels.channels = [Channel(samples_per_pixel = samples_per_pixel, \
                                       id = ChannelID("Channel:0:{0}".format(index))) \
                               for index in range(c_count)]

        if c_count > 1:
            if self.has_s_axis:
                for index in range(c_count):
                    ome_pixels.channels[index].color = ome_rgb_colors[index % len(ome_rgb_colors)]
            else:
                for index in range(c_count):
                    ome_pixels.channels[index].color = ome_multi_colors[index % len(ome_multi_colors)]
        else:
            ome_pixels.channels[0].color = ome_grayscale

        ome_image = Image(name = filename, id = "Image:0", pixels = ome_pixels)
        ome_xml = to_xml(OME(images = [ome_image])).encode()

        with open(filename, "wb") as fileio:
            with tifffile.TiffWriter(fileio, bigtiff = bigtiff) as tiff:
                tiff.write(output_array, description = ome_xml, metadata = None)

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

        if tiff.is_micromanager:
            try:
                summary = tiff.micromanager_metadata['Summary']
                logger.debug('Reading micromanager metadata: {0}'.format(summary))
                metadata['x_pixel_um'] = summary.get('PixelSize_um', default_pixel_um)
                metadata['y_pixel_um'] = summary.get('PixelSize_um', default_pixel_um)
                metadata['z_step_um'] = summary.get('z-step_um', default_z_step_um)

                settings = json.loads(summary.get('SPIMAcqSettings', ''))
                if settings.get('useTimePoints', False) == True:
                    metadata['finterval_sec'] = settings.get('timePointInterval', default_finterval_sec)
                else:
                    metadata['finterval_sec'] = settings.get('sliceDuration', default_finterval_sec)
            
                return metadata

            except Exception as e:
                logger.warning("Failed to load micromanager metadata. Try loading OME metadata.")
                logger.debug(e)

        if tiff.ome_metadata is not None:
            try:
                ome = from_xml(tiff.ome_metadata)
                logger.debug('Reading ome metadata: {0}'.format(ome.images[series]))
                if hasattr(ome.images[series].pixels, "physical_size_x"):
                    ratio = ome_ratio_to_um.get(ome.images[series].pixels.physical_size_x_unit, 1.0)
                    metadata['x_pixel_um'] = ome.images[series].pixels.physical_size_x * ratio
                else:
                    metadata['x_pixel_um'] = default_pixel_um

                if hasattr(ome.images[series].pixels, "physical_size_y"):
                    ratio = ome_ratio_to_um.get(ome.images[series].pixels.physical_size_y_unit, 1.0)
                    metadata['y_pixel_um'] = ome.images[series].pixels.physical_size_y * ratio
                else:
                    metadata['y_pixel_um'] = default_pixel_um

                if hasattr(ome.images[series].pixels, "physical_size_z"):
                    ratio = ome_ratio_to_um.get(ome.images[series].pixels.physical_size_z_unit, 1.0)
                    metadata['z_step_um'] = ome.images[series].pixels.physical_size_z * ratio
                else:
                    metadata['z_step_um'] = default_z_step_um

                if hasattr(ome.images[series].pixels, "time_increment"):
                    ratio = ome_ratio_to_sec.get(ome.images[series].pixels.time_increment_unit, 1.0)
                    metadata['finterval_sec'] = ome.images[series].pixels.time_increment * ratio
                else:
                    metadata['finterval_sec'] = default_finterval_sec

                return metadata

            except Exception as e:
                logger.warning("Failed to load ome-tiff metadata. Try loading ImageJ metadata.")
                logger.debug(e)

        if tiff.imagej_metadata is not None:
            logger.debug('Reading imagej metadata: {0}'.format(tiff.imagej_metadata))
            ratio = imagej_ratio_to_um.get(tiff.imagej_metadata.get('unit', 'um'), 1.0)
            metadata['z_step_um'] = tiff.imagej_metadata.get('spacing', default_z_step_um / ratio) * ratio
            metadata['finterval_sec'] = tiff.imagej_metadata.get('finterval_sec', default_finterval_sec)
        else:
            logger.debug('No imagej metadata found')
            metadata['z_step_um'] = default_z_step_um
            metadata['finterval_sec'] = default_finterval_sec
            if 'ResolutionUnit' in tiff.pages[series].tags:
                ratio = resunit_ratio_to_um.get(tiff.pages[series].tags['ResolutionUnit'].value, 1.0)
            else:
                ratio = 1.0

        if 'XResolution' in tiff.pages[series].tags:
            values = tiff.pages[series].tags['XResolution'].value
            metadata['x_pixel_um'] = float(values[1]) / float(values[0]) * ratio
        else:
            metadata['x_pixel_um'] = default_pixel_um

        if 'YResolution' in tiff.pages[series].tags:
            values = tiff.pages[series].tags['YResolution'].value
            metadata['y_pixel_um'] = float(values[1]) / float(values[0]) * ratio
        else:
            metadata['y_pixel_um'] = default_pixel_um

        return metadata

    def __set_metadata (self, metadata):
        self.voxel_um = [metadata['z_step_um'], metadata['y_pixel_um'], metadata['x_pixel_um']]
        self.finterval_sec = metadata['finterval_sec']

    def resize_image (self, shape, centering = False, offset = None):
        shape = [self.t_count, self.c_count] + list(shape)
        if offset is not None:
            offset = [0, 0] + list(offset)
        self.resize_all(shape, centering = centering, offset = offset)

    def resize_all (self, shape, centering = False, offset = None):
        shape = list(shape) + self.image_array.shape[len(shape):]
        if offset is not None:
            offset = list(offset) + [0] * (len(self.image_array) - len(offset))

        slices_src, slices_tgt = gpuimage.pasting_slices(self.image_array.shape, shape, centering = centering, offset = offset)
        new_array = np.zeros(shape, dtype = self.image_array.dtype)
        new_array[tuple(slices_tgt)] = self.image_array[tuple(slices_src)].copy()

        self.image_array = new_array
        self.update_dimensions()

    def crop_image (self, origin, shape):
        origin = [0, 0] + list(origin)
        shape = [self.t_count, self.c_count] + list(shape)
        self.crop_all(origin, shape)

    def crop_all (self, origin, shape):
        slice_list = [slice(o, s, 1) for o, s in zip(origin, shape)]
        self.crop_by_slice(slice_list)

    def crop_by_slice (self, slice_list):
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
                        image = image_func(image, t_index, c_index)
                        output_images.append(image)
                    output_channels.append(output_images)
                else:
                    image = self.image_array[t_index, c_index]
                    image = image_func(image, t_index, c_index)
                    output_channels.append(image)
            output_frames.append(output_channels)
            yield t_index

        self.image_array = np.array(output_frames)
        self.update_dimensions()

    def apply_all (self, image_func, progress = False):
        if progress:
            with ProgressBar(max_value = self.t_count) as bar:
                for index in self.__apply_all(image_func):
                    bar.update(index + 1)
        else:
            for index in self.__apply_all(image_func):
                pass

    def scale_by_ratio (self, ratio = 1.0, gpu_id = None):
        ratio = gpuimage.expand_ratio(ratio)
        def scale_func (image, t_index, c_index):
            return gpuimage.scale(image, ratio, gpu_id = gpu_id)
        self.apply_all(scale_func)
        self.voxel_um = [self.voxel_um[i] / ratio[i] for i in range(len(self.voxel_um))]

    def scale_by_pixelsize (self, pixel_um, gpu_id = None):
        pixel_um = gpuimage.expand_ratio(pixel_um)
        ratio = [self.voxel_um[i] / pixel_um[i] for i in range(len(self.voxel_um))]
        self.scale_by_ratio(ratio = ratio, gpu_id = gpu_id)

    def scale_isometric (self, gpu_id = None):
        if np.isclose(self.voxel_um[1], self.voxel_um[2]) == False:
            logger.warning("X and Y pixel size are different: {0}".format(self.voxel_um))

        pixel_um = min(self.voxel_um)
        self.scale_by_pixelsize(pixel_um, gpu_id = gpu_id)

    def rotate (self, angle = 0.0, axis = 0, gpu_id = None):
        rot_tuple = gpuimage.axis_to_tuple(axis)
        def rotate_func (image, t_index, c_index):
            return gpuimage.rotate(image, angle, rot_tuple, gpu_id = gpu_id)
        self.apply_all(rotate_func)

    def affine_transform (self, matrix, gpu_id = None):
        def affine_func (image, t_index, c_index):
            return gpuimage.affine_transform(image, matrix, gpu_id = gpu_id)
        self.apply_all(affine_func)

    def shift (self, offset, gpu_id = None):
        def shift_func (image, t_index, c_index):
            return gpuimage.shift(image, offset, gpu_id = gpu_id)
        self.apply_all(shift_func)
