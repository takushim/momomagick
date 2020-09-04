#!/usr/bin/env python

import os, platform, sys, glob, argparse
import numpy, pandas, tifffile
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess

align_table = pandas.read_csv("align2.txt", comment = '#', sep = '\t')

align_plane = numpy.array(align_table.align_plane)
align_x = numpy.array(align_table.align_x)
align_y = numpy.array(align_table.align_y)

smooth_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
smooth_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)


pyplot.plot(align_plane, align_x)
pyplot.plot(align_plane, align_y)
pyplot.plot(align_plane, smooth_x)
pyplot.plot(align_plane, smooth_y)
pyplot.show()


