#!/usr/bin/env python

import sys, numpy, pandas, time
from scipy.optimize import curve_fit

class Lifetime:
    def __init__ (self, spot_table, time_scale = 1.0):
        self.lifetime_columns = ['life_count', 'life_time', 'spot_count']
        self.time_scale = time_scale
        self.spot_table = spot_table

    @staticmethod
    def add_life_count (spot_table):
        work_table = spot_table.copy()

        work_table['life_count'] = (work_table.groupby('total_index').cumcount())
        lifetime_table = work_table['total_index'].value_counts().to_frame('life_total')
        work_table = pandas.merge(work_table, lifetime_table, \
                                  left_on='total_index', right_index=True, how='left')

        return work_table

    @staticmethod
    def fit_one_phase_decay (times, counts):
        def one_phase_decay (x, a, b, c):
            return a * numpy.exp(- b * x) + c

        popt, pcov = curve_fit(one_phase_decay, times, counts)

        return (lambda x: popt[0] * numpy.exp (- popt[1] * x) + popt[2]), popt, pcov

    def regression (self):
        # regression
        work_table = self.spot_table.copy()

        # spots to be counted
        index_set = set(work_table[work_table.plane == 1].total_index.tolist())
        print("Regression set:", index_set)

        # regression
        output_indexes = []
        output_counts = []
        for index in range(0, work_table.plane.max() + 1):
            spot_count = len(work_table[(work_table.total_index.isin(index_set)) & (work_table.plane == (index + 1))])
            output_indexes += [index]
            output_counts += [spot_count]

        # output data
        output_times = [i * self.time_scale for i in output_indexes]
        output_table = pandas.DataFrame({ \
                            self.lifetime_columns[0] : output_indexes, \
                            self.lifetime_columns[1] : output_times, \
                            self.lifetime_columns[2] : output_counts}, \
                            columns = self.lifetime_columns)

        return output_table
    
    def lifetime (self):
        # add lifetime columns
        work_table = self.add_life_count(self.spot_table)

        # prepare data
        lifecount_max = work_table.life_total.max()
        work_table = work_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)
        output_indexes = [i for i in range(1, lifecount_max + 1)]
        output_times = [i * self.time_scale for i in output_indexes]
        output_counts = [len(work_table[work_table.life_count == i]) for i in output_indexes]

        # output data
        output_table = pandas.DataFrame({ \
                            self.lifetime_columns[0] : output_indexes, \
                            self.lifetime_columns[1] : output_times, \
                            self.lifetime_columns[2] : output_counts}, \
                            columns = self.lifetime_columns)

        return output_table
    
    def cumulative (self):
        # add lifetime columns
        work_table = self.add_life_count(self.spot_table)

        # prepare data
        lifecount_max = work_table.life_total.max()
        work_table = work_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)
        output_indexes = [i for i in range(1, lifecount_max + 1)]
        output_times = [i * self.time_scale for i in output_indexes]
        output_counts = [len(work_table[work_table.life_count >= i]) for i in output_indexes]

        # output data
        output_table = pandas.DataFrame({ \
                            self.lifetime_columns[0] : output_indexes, \
                            self.lifetime_columns[1] : output_times, \
                            self.lifetime_columns[2] : output_counts}, \
                            columns = self.lifetime_columns)

        return output_table

    def newbinding (self):
        # add lifetime columns
        work_table = self.add_life_count(self.spot_table)
        
        # prepare data
        work_table = work_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        work_table['life_time'] = work_table['life_total'] * self.time_scale

        return work_table[['plane', 'life_total', 'life_time']]

    ### These functions are recorded as a bad but important hack (now being used)
    @staticmethod
    def filter_spots_maskimage (spot_table, mask_image):
        first_spot_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        first_spot_table['mask'] = mask_image[first_spot_table.y.values.astype(numpy.int), first_spot_table.x.values.astype(numpy.int)]
        index_set = set(first_spot_table[first_spot_table['mask'] > 0].total_index.to_list())
        return spot_table[spot_table.total_index.isin(index_set)]

    ### These functions are recorded as a bad but important hack
    def filter_spots_lifetime (self, spot_table, lifetime_min = 0, lifetime_max = numpy.inf):
        spot_table = spot_table.sort_values(by = ['total_index', 'plane']).reset_index(drop=True)
        spot_table = spot_table[(lifetime_min <= spot_table.life_total) & \
                                (spot_table.life_total <= lifetime_max)].reset_index(drop=True)

        return spot_table

    def omit_lastplane_spots (self, spot_table, lastplane_index):
        total_indexes = spot_table[spot_table.plane == lastplane_index].total_index.tolist()
        total_indexes = list(set(total_indexes))

        return spot_table[~spot_table.total_index.isin(total_indexes)]

    def keep_first_spots (self, spot_table):
        spot_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        return spot_table

    def keep_last_spots (self, spot_table):
        spot_table = spot_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)
        return spot_table

    def average_spots (self, spot_table):
        agg_dict = {x : numpy.max for x in spot_table.columns}
        agg_dict['x'] = numpy.mean
        agg_dict['y'] = numpy.mean
        agg_dict['intensity'] = numpy.mean
        agg_dict['distance'] = numpy.sum
        spot_table = spot_table.groupby('total_index').agg(agg_dict).reset_index(drop=True)
        return spot_table


