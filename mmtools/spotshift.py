#!/usr/bin/env python

import sys, numpy, pandas, time
from statsmodels.nonparametric.smoothers_lowess import lowess

class SpotShift:
    def __init__ (self, spot_scaling = 1.0, spot_shift = [0.0, 0.0]):
        self.shift = spot_shift
        self.scaling = spot_scaling

    def shift_spots (self, spot_table):
        work_table = spot_table.copy()

        # save original spots and parameters
        work_table['orig_x'] = work_table['x']
        work_table['orig_y'] = work_table['y']
        work_table['shift_x'] = self.shift[0]
        work_table['shift_y'] = self.shift[1]
        work_table['scaling'] = self.scaling

        # shift spots
        print("Scaling: {0:f}, shift: ({1:f}, {2:f}), ".format(self.scaling, self.shift[0], self.shift[1]))
        work_table['moved_x'] = work_table['x'] * work_table['scaling'] + work_table['shift_x']
        work_table['moved_y'] = work_table['y'] * work_table['scaling'] + work_table['shift_y']

        # update coordinates
        work_table['x'] = work_table['moved_x']
        work_table['y'] = work_table['moved_y']

        return work_table

    @staticmethod
    def add_smoothing (align_table, fraction = 0.1):
        work_table = align_table.copy()

        # lowess algorithm
        smooth_x = lowess(work_table.align_x, work_table.align_plane, frac = fraction, return_sorted = False)
        smooth_y = lowess(work_table.align_y, work_table.align_plane, frac = fraction, return_sorted = False)

        # adjust the alignment at t = 0 as (0, 0)
        smooth_x = smooth_x - smooth_x[0]
        smooth_y = smooth_y - smooth_y[0]

        work_table['smooth_x'] = smooth_x
        work_table['smooth_y'] = smooth_y
        return work_table

    @staticmethod
    def align_spots (spot_table, align_table, use_smoothing = False, force_calc_smoothing = False):
        # add smoothing
        if use_smoothing:
            if (not {'smooth_x', 'smooth_y'} <= set(align_table.columns)) or force_calc_smoothing:
                print("Calculating smoothing. Smoothing data in the input file are ignored.")
                align_table = SpotShift.add_smoothing(align_table)

        # alignment
        work_table = pandas.merge(spot_table, align_table, left_on='plane', right_on='align_plane', how='left')
        if use_smoothing:
            work_table['aligned_x'] = work_table['x'] - work_table['smooth_x']
            work_table['aligned_y'] = work_table['y'] - work_table['smooth_y']
        else:
            work_table['aligned_x'] = work_table['x'] - work_table['align_x']
            work_table['aligned_y'] = work_table['y'] - work_table['align_y']
        
        # copy coodinates
        work_table['x'] = work_table['aligned_x']
        work_table['y'] = work_table['aligned_y']

        return work_table

