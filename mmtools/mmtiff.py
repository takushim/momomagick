#!/usr/bin/env python

import sys, platform, re, tifffile
import numpy as np
from pathlib import Path

def stem (filename):
    name = Path(filename).stem
    name = re.sub('\.ome$', '', name, flags=re.IGNORECASE)
    return name

def prefix (filename):
    name = Path(filename).stem
    name = re.sub('\.ome$', '', name, flags=re.IGNORECASE)
    name = re.sub('MMStack_Pos[0-9]+$', '', name, flags=re.IGNORECASE)
    name = re.sub('_$', '', name, flags=re.IGNORECASE)
    return name

def with_suffix (filename, suffix):
    name = stem(filename)
    if name == name + suffix:
        raise Exception('Empty suffix. May overwrite the original file. Exiting.')
    return name + suffix

def font_path ():
    if platform.system() == "Windows":
        font_filename = 'C:/Windows/Fonts/Arial.ttf'
    elif platform.system() == "Linux":
        font_filename = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    elif platform.system() == "Darwin":
        font_filename = '/Library/Fonts/Verdana.ttf'
    else:
        raise Exception('Font file error.')

    return font_filename

def float_to_int (image_array, dtype = np.uint16):
    return np.clip(image_array, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

def convert_to_uint8 (image_array):
    max_value = image_array.max()
    min_value = image_array.min()
    return (image_array / (max_value - min_value) * 255.0).astype(np.uint8)

def paste_slices (src_shape, tgt_shape, center = False):
    if center:
        shifts = (np.array(tgt_shape) - np.array(src_shape)) // 2
    else:
        shifts = np.array([0 for i in range(len(src_shape))])

    src_starts = [min(-x, y) if x < 0 else 0 for x, y in zip(shifts, src_shape)]
    src_bounds = np.minimum(src_shape, tgt_shape - shifts)
    slices_src = tuple([slice(x, y) for x, y in zip(src_starts, src_bounds)])

    tgt_starts = [0 if x < 0 else min(x, y) for x, y in zip(shifts, tgt_shape)]
    tgt_bounds = np.minimum(src_shape + shifts, tgt_shape)
    slices_tgt = tuple([slice(x, y) for x, y in zip(tgt_starts, tgt_bounds)])

    return [slices_src, slices_tgt]

def resize (image_array, shape, center = False):
    resized_array = np.zeros(shape, dtype = image_array.dtype)
    slices_src, slices_tgt = paste_slices(image_array.shape, shape, center)
    resized_array[slices_tgt] = image_array[slices_src].copy()
    return resized_array

class MMTiff:
    def __init__ (self, filename):
        # set default values
        self.micromanager_summary = None
        self.pixelsize_um = 0.1625
        self.z_step_um = 0.5

        # read image
        self.filename = filename
        self.read_image()

    def read_image (self):
        # read TIFF file (assumes TZ(C)YX(S) order)
        print("Reading image:", self.filename)
        self.image_list = []
        with tifffile.TiffFile(self.filename) as tiff:
            axes = tiff.series[0].axes
            image = tiff.asarray(series = 0)
            if 'T' in axes:
                self.total_time = len(image)
                self.image_list = [x[0] for x in np.split(image, len(image))] # remove the first axis [1, (frame,) height, width]
            else:
                self.total_time = 1
                self.image_list = [image]
            print('Load image:', image.shape, axes)

            #self.micromanager_summary = None
            #self.set_metadata()
            if tiff.is_micromanager:
                self.read_micromanager_metadata(tiff)
            else:
                self.read_image_metadata(tiff)

        if 'Z' not in axes:
            for index in range(len(self.image_list)):
                self.image_list[index] = self.image_list[index][np.newaxis]
        if 'C' not in axes:
            for index in range(len(self.image_list)):
                self.image_list[index] = self.image_list[index][:, np.newaxis]

        self.total_zstack = self.image_list[0].shape[0]
        self.total_channel = self.image_list[0].shape[1]
        self.height = self.image_list[0].shape[2]
        self.width = self.image_list[0].shape[3]
        if 'S' in axes:
            self.axes = 'TZCYXS'
            self.colored = True
        else:
            self.axes = 'TZCYX'
            self.colored = False
        self.dtype = self.image_list[0].dtype

        print('Original image was shaped into: ', self.total_time, ' x ', self.image_list[0].shape, self.axes)

    def read_micromanager_metadata (self, tiff):
        self.micromanager_summary = tiff.micromanager_metadata['Summary']
        self.pixelsize_um = float(self.micromanager_summary['PixelSize_um'])
        self.z_step_um = float(self.micromanager_summary['z-step_um'])
    
    def read_image_metadata (self, tiff):
        if 'XResolution' in tiff.pages[0].tags:
            values = tiff.pages[0].tags['XResolution'].value
            self.pixelsize_um = float(values[1]) / float(values[0])
            print("Set pixelsize_um from the image", self.pixelsize_um)
        if 'ImageDescription' in tiff.pages[0].tags:
            if tiff.imagej_metadata is not None:
                if 'spacing' in tiff.imagej_metadata:
                    self.z_step_um = tiff.imagej_metadata['spacing']
                    print("Set z_step_um from the imagej metadata", self.z_step_um)
            elif tiff.ome_metadata is not None:
                if 'spacing' in tiff.ome_metadata:
                    self.z_step_um = tiff.ome_metadata['spacing']
                    print("Set z_step_um from the ome metadata", self.z_step_um)

    def as_list (self, channel = None, drop = True, list_channel = False):
        if channel is None:
            if list_channel:
                return [[x[:, i] for i in range(self.total_channel)] for x in self.image_list]
            else:
                return self.image_list
        else:
            if drop is True:
                print("Using channel (dropping channel):", channel)
                return [x[:, channel] for x in self.image_list]
            else:
                print("Using channel (keeping channel):", channel)
                if list_channel:
                    return [[x[:, channel]] for x in self.image_list]
                else:
                    return [x[:, channel:(channel + 1)] for x in self.image_list]

    def as_array (self, channel = None, drop = True, channel_first = False):
        if channel is None:
            image_array = np.array(self.image_list)
            if channel_first:
                return image_array.swapaxes(1, 2)
            else:
                return image_array
        else:
            if drop is True:
                print("Using channel (dropping channel):", channel)
                return np.array([x[:, channel] for x in self.image_list])
            else:
                print("Using channel (keeping channel):", channel)
                image_array = np.array([x[:, channel:(channel + 1)] for x in self.image_list])
                if channel_first:
                    image_array.swapaxes(1, 2)
                else:
                    return image_array

    def save_image (self, filename, image_array):
        print('Saving image: ', image_array.shape, image_array.dtype)
        metadata = {'spacing': self.z_step_um, 'unit': 'um', 'Composite mode': 'composite'}
        tifffile.imsave(filename, np.array(image_array), imagej = True, \
                resolution = (1 / self.pixelsize_um, 1 / self.pixelsize_um), \
                metadata = metadata)

    def save_image_ome (self, filename, image_array):
        print('Saving image: ', image_array.shape, image_array.dtype)
        #metadata = {'PhysicalSizeX': self.pixelsize_um, 'PhysicalSizeXUnit': 'um', \
        #            'PhysicalSizeY': self.pixelsize_um, 'PhysicalSizeYUnit': 'um', \
        #            'PhysicalSizeZ': self.z_step_um, 'PhysicalSizeZUnit': 'um'}
        metadata = {'spacing': self.z_step_um, 'unit': 'um', 'Composite mode': 'composite'}
        tifffile.imsave(filename, np.array(image_array), ome = True, \
                        resolution = (1 / self.pixelsize_um, 1 / self.pixelsize_um), \
                        metadata = metadata)
