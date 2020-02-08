#!/usr/bin/env python

import sys, numpy
from skimage.external import tifffile

class MMTiff:
    def __init__ (self, filename):
        self.filename = filename
        self.axes = None
        self.image_list = []
        self.metadata = None
        self.total_time = 0
        self.total_channel = 1
        self.total_zstack = 1
        self.width = 0
        self.height = 0
        self.colored = False

        # read TIFF file (assumes TZ(C)YX(S) order)
        self.image_list = []
        with tifffile.TiffFile(filename) as tiff:
            axes = tiff.series[0].axes
            image = tiff.asarray(series = 0)
            if 'T' in axes:
                self.total_time = len(image)
                self.image_list = [x[0] for x in numpy.split(image, len(image))] # remove the first axis [1, (frame,) height, width]
            else:
                self.total_time = 1
                self.image_list = [image]
            print('Image shape:', self.total_time, ' x ', image.shape)
            print('Image axes:', axes)

        if 'Z' not in axes:
            print('Adding temporary Z axis.')
            for index in range(len(self.image_list)):
                self.image_list[index] = self.image_list[index][numpy.newaxis]
        if 'C' not in axes:
            print('Adding temporary C axis.')
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

        #print(vars(self))
        print('Original image was shaped into: ', self.axes, self.total_time, ' x ', self.image_list[0].shape)


