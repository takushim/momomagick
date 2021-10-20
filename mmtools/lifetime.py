#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

life_columns = ['frame', 'time', 'spotcount']
binding_columns = ['plane', 'lifeframe', 'lifetime']

def life_table (count_list, time_scale = 1.0):
    frame_list = [(i + 1) for i in range(len(count_list))]
    time_list = [(i + 1) * time_scale for i in range(len(count_list))]
    table = pd.DataFrame({life_columns[0]: frame_list,
                          life_columns[1]: time_list,
                          life_columns[2]: count_list},
                          columns = life_columns)
    return table

def binding_table (plane_list, count_list, time_scale = 1.0):
    count_list = [(i + 1) for i in count_list]
    time_list = [i * time_scale for i in count_list]
    table = pd.DataFrame({binding_columns[0]: plane_list,
                          binding_columns[1]: count_list,
                          binding_columns[2]: time_list},
                          columns = binding_columns)
    return table

def life_count (spot_table):
    return spot_table.groupby('total_index').cumcount().to_list()

def fit_one_phase_decay (time_list, count_list, start = 0):
    if start > 0:
        print("Start fitting from:", start)
        time_list = time_list[start:]
        count_list = count_list[start:]

    def one_phase_decay (x, a, b, c):
        return (a - c) * np.exp(- b * x) + c
    
    popt, pcov = curve_fit(one_phase_decay, time_list, count_list)
    result = {'func': lambda x: popt[0] * np.exp (- popt[1] * x) + popt[2],
              'popt': popt,
              'pcov': pcov,
              'halflife': np.log(2) / popt[1],
              'koff': popt[1],
              'start': start
    }
    return result

def regression (spot_table, time_scale = 1.0):
    # regression
    work_table = spot_table.copy()
    start_plane = np.min(work_table.plane)
    if start_plane > 0:
        print("Setting plane {0} as the beginning".format(start_plane))
        work_table.plane = work_table.plane - start_plane

    # spots to be counted
    index_set = set(work_table[work_table.plane == 0].total_index.tolist())
    print("Regression set:", index_set)

    # regression
    output_counts = []
    for index in range(work_table.plane.max() + 1):
        spot_count = len(work_table[(work_table.total_index.isin(index_set)) & (work_table.plane == index)])
        output_counts.append(spot_count)

    return life_table(output_counts, time_scale = time_scale)

def lifetime (spot_table, time_scale = 1.0):
    # add lifetime
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # drop plane starting from the time-lapse image
    start_plane = np.min(work_table.plane)
    index_set = set(work_table[work_table.plane == start_plane].total_index.tolist())
    print("Dropping spots that start from plane {0}:".format(start_plane), index_set)
    work_table = work_table[work_table.total_index.isin(index_set) == False]

    # prepare data (life_count starts from 0)
    work_table = work_table.drop_duplicates(subset = 'total_index', keep = 'last').reset_index(drop = True)
    life_max = work_table.life_count.max() + 1
    output_counts = [len(work_table[work_table.life_count == i]) for i in range(life_max)]

    return life_table(output_counts, time_scale = time_scale)

def cumulative (spot_table, time_scale = 1.0):
    # add lifetime columns
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # drop plane starting from the time-lapse image
    start_plane = np.min(work_table.plane)
    index_set = set(work_table[work_table.plane == start_plane].total_index.tolist())
    print("Dropping spots that start from plane {0}:".format(start_plane), index_set)
    work_table = work_table[work_table.total_index.isin(index_set) == False]

    # prepare data (life_count starts from 0)
    life_max = work_table.life_count.max() + 1
    work_table = work_table.drop_duplicates(subset = 'total_index', keep = 'last').reset_index(drop = True)
    output_counts = [len(work_table[work_table.life_count >= i]) for i in range(life_max)]

    return life_table(output_counts, time_scale = time_scale)

def new_bindings (spot_table, time_scale = 1.0):
    # add lifetime columns
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # prepare data (life_count starts from 0)
    agg_table = {'plane': np.min, 'life_count': np.max}
    work_table = work_table.groupby('total_index').agg(agg_table)
    
    plane_list = work_table['plane'].to_list()
    maxcount_list = work_table['life_count'].to_list()

    return binding_table(plane_list, maxcount_list, time_scale)

# This is for spotmarker
def filter_spots_maskimage (spot_table, mask_image):
    first_spot_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
    first_spot_table['mask'] = mask_image[first_spot_table.y.values.astype(np.int), first_spot_table.x.values.astype(np.int)]
    index_set = set(first_spot_table[first_spot_table['mask'] > 0].total_index.to_list())
    return spot_table[spot_table.total_index.isin(index_set)]
