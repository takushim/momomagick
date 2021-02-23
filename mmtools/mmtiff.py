#!/usr/bin/env python

import platform, sys, numpy, pathlib, re, tifffile

class MMTiff:
    def __init__ (self, filename):
        # set default values
        self.micromanager_summary = None
        self.pixelsize_um = 0.1625
        self.z_step_um = 0.5

        # read image
        self.filename = filename
        self.read_image()

    @staticmethod
    def stem (filename):
        name = pathlib.Path(filename).stem
        name = re.sub('\.ome$', '', name, flags=re.IGNORECASE)
        return name

    @staticmethod
    def prefix (filename):
        name = pathlib.Path(filename).stem
        name = re.sub('\.ome$', '', name, flags=re.IGNORECASE)
        name = re.sub('MMStack_Pos[0-9]+$', '', name, flags=re.IGNORECASE)
        name = re.sub('_$', '', name, flags=re.IGNORECASE)
        return name

    @staticmethod
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

    @staticmethod
    def float_to_int (image_array, dtype = numpy.uint16):
        print("Using dtype:", dtype)
        range_max = numpy.iinfo(dtype).max
        image_max = numpy.max(image_array)
        image_min = numpy.min(image_array)
        return (image_array * (image_max - image_min) / image_max).astype(dtype)

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
                self.read_micromanager_metadata(tiff)
            else:
                self.read_image_metadata(tiff)

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
            self.z_step_um = tiff.imagej_metadata['spacing']
            print("Set z_step_um from the image", self.z_step_um)

    def as_list (self, channel = None, drop = True):
        if channel is None:
            return self.image_list
        else:
            if drop is True:
                print("Using channel (dropping channel):", channel)
                return [x[:, channel] for x in self.image_list]
            else:
                print("Using channel (keeping channel):", channel)
                return [x[:, channel:(channel + 1)] for x in self.image_list]

    def as_array (self, channel = None, drop = True):
        if channel is None:
            return numpy.array(self.image_list)
        else:
            if drop is True:
                print("Using channel (dropping channel):", channel)
                return numpy.array([x[:, channel] for x in self.image_list])
            else:
                print("Using channel (keeping channel):", channel)
                return numpy.array([x[:, channel:(channel + 1)] for x in self.image_list])

    def save_image (self, filename, image_array):
        print('Saving image: ', image_array.shape, image_array.dtype)
        tifffile.imsave(filename, numpy.array(image_array), imagej = True, \
                resolution = (1 / self.pixelsize_um, 1 / self.pixelsize_um), \
                metadata = {'spacing': self.z_step_um, 'unit': 'um', 'Composite mode': 'composite'})

