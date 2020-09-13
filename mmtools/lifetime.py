#!/usr/bin/env python

import sys, numpy, pandas, time

class Lifetime:
    def __init__ (self):
        self.lifetime_columns = ['life_count', 'life_time', 'spot_count']

    def add_life_count (self, spot_table):
        work_table = spot_table.copy()

        work_table['life_count'] = (work_table.groupby('total_index').cumcount())
        lifetime_table = work_table['total_index'].value_counts().to_frame('life_total')
        work_table = pandas.merge(work_table, lifetime_table, \
                                  left_on='total_index', right_index=True, how='left')

        return work_table

    def regression (self, spot_table, time_scale = 1.0):
        # regression
        work_table = spot_table.copy()

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
        output_times = [i * time_scale for i in output_indexes]
        output_table = pandas.DataFrame({ \
                            self.lifetime_columns[0] : output_indexes, \
                            self.lifetime_columns[1] : output_times, \
                            self.lifetime_columns[2] : output_counts}, \
                            columns = self.lifetime_columns)

        return output_table
    
    def lifetime (self, spot_table, time_scale = 1.0):
        # add lifetime columns
        work_table = self.add_life_count(spot_table)

        # prepare data
        lifecount_max = work_table.life_total.max()
        work_table = work_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)
        output_indexes = [i for i in range(1, lifecount_max + 1)]
        output_times = [i * time_scale for i in output_indexes]
        output_counts = [len(work_table[work_table.life_count == i]) for i in output_indexes]

        # output data
        output_table = pandas.DataFrame({ \
                            self.lifetime_columns[0] : output_indexes, \
                            self.lifetime_columns[1] : output_times, \
                            self.lifetime_columns[2] : output_counts}, \
                            columns = self.lifetime_columns)

        return output_table
    
    def new_binding (self, spot_table, time_scale = 1.0):
        # add lifetime columns
        work_table = self.add_life_count(spot_table)
        
        # prepare data
        work_table = work_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        work_table['life_time'] = work_table['life_total'] * time_scale

        return work_table[['plane', 'life_total', 'life_time']]


