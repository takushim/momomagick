#!/usr/bin/env python

import sys, numpy, pathlib, re, tifffile

class MMTiff:
    def __init__ (self, filename):
        self.filename = filename
        self.read_image()

    @staticmethod
    def filename_stem (filename):
        stem = pathlib.Path(filename).stem
        stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)

        return stem

    def read_image (self):
        # read TIFF file (assumes TZ(C)YX(S) order)
        print("Reading image:", self.filename)
        self.image_list = []
        with tifffile.TiffFile(self.filename) as tiff:
            axes = tiff.series[0].axes
            image = tiff.asarray(series = 0)
            if 'T' in axes:
                self.total_time = len(image)
                self.image_list = [x[0] for x in numpy.split(image, len(image))] # remove the first axis [1, (frame,) height, width]
            else:
                self.total_time = 1
                self.image_list = [image]
            print('Load image:', image.shape, axes)

            #self.micromanager_summary = None
            #self.set_metadata()
            if tiff.is_micromanager:
                self.micromanager_summary = tiff.micromanager_metadata['Summary']
            else:
                self.micromanager_summary = None
            self.set_metadata()

        if 'Z' not in axes:
            for index in range(len(self.image_list)):
                self.image_list[index] = self.image_list[index][numpy.newaxis]
        if 'C' not in axes:
            for index in range(len(self.image_list)):
                self.image_list[index] = self.image_list[index][:, numpy.newaxis]

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

    def set_metadata (self):
        if self.micromanager_summary:
            self.starttime = self.micromanager_summary['StartTime']
            self.exposure_ms = float(self.micromanager_summary['LaserExposure_ms'])
            self.pixelsize_um = float(self.micromanager_summary['PixelSize_um'])
            self.z_step_um = float(self.micromanager_summary['z-step_um'])
        else:
            self.starttime = '2019-11-08 11:09:13 -0500'
            self.exposure_ms = 1000.0
            self.pixelsize_um = 0.1625
            self.z_step_um = 0.5
            print("No micromanager summary. Setting default values.")

    def as_list (self, channel = None, drop_channel = True):
        if channel is None:
            return self.image_list
        else:
            if drop_channel is True:
                print("Using channel (dropping channel):", channel)
                return [x[:, channel] for x in self.image_list]
            else:
                print("Using channel (keeping channel):", channel)
                return [x[:, channel:(channel + 1)] for x in self.image_list]

    def as_array (self, channel = None, drop_channel = True):
        if channel is None:
            return numpy.array(self.image_list)
        else:
            if drop_channel is True:
                print("Using channel (dropping channel):", channel)
                return numpy.array([x[:, channel] for x in self.image_list])
            else:
                print("Using channel (keeping channel):", channel)
                return numpy.array([x[:, channel:(channel + 1)] for x in self.image_list])
    
    def save_image (self, filename, image_array):
        print('Saving image: ', image_array.shape)
        tifffile.imsave(filename, numpy.array(image_array), imagej = True, \
                resolution = (1 / self.pixelsize_um, 1 / self.pixelsize_um), \
                metadata = {'spacing': self.z_step_um, 'unit': 'um', 'Composite mode': 'composite'})

