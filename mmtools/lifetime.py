#!/usr/bin/env python

import numpy as np
import pandas as pd
from logging import getLogger
from scipy import optimize

logger = getLogger(__name__)

life_columns = ['frame', 'time', 'spotcount']
binding_columns = ['plane', 'lifeframe', 'lifetime']

optimizing_methods = ["Powell", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B", "SLSQP"]
default_method = "Nelder-Mead"

def life_table (count_list, time_scale = 1.0):
    if (count_list is None) or (len(count_list) == 0):
        table = pd.DataFrame({life_columns[0]: pd.Series(dtype = int),
                              life_columns[1]: pd.Series(dtype = float),
                              life_columns[2]: pd.Series(dtype = int)},
                              columns = life_columns)
    else:
        frame_list = [(i + 1) for i in range(len(count_list))]
        time_list = [(i + 1) * time_scale for i in range(len(count_list))]
        table = pd.DataFrame({life_columns[0]: frame_list,
                              life_columns[1]: time_list,
                              life_columns[2]: count_list},
                              columns = life_columns)
    return table

def binding_table (plane_list, count_list, time_scale = 1.0):
    if (count_list is None) or (len(count_list) == 0) or \
       (plane_list is None) or (len(plane_list) == 0):
        table = pd.DataFrame({binding_columns[0]: pd.Series(dtype = int),
                              binding_columns[1]: pd.Series(dtype = int),
                              binding_columns[2]: pd.Series(dtype = float)},
                              columns = binding_columns)
    else:
        count_list = [(i + 1) for i in count_list]
        time_list = [i * time_scale for i in count_list]
        table = pd.DataFrame({binding_columns[0]: plane_list,
                              binding_columns[1]: count_list,
                              binding_columns[2]: time_list},
                              columns = binding_columns)
    return table

def life_count (spot_table):
    return spot_table.groupby('total_index').cumcount().to_list()

def fit_one_phase_decay (time_list, count_list, start = 0, end = 0, method = default_method):
    if len(count_list) == 0 or len(time_list) == 0:
        result = {
            'func': lambda x: 1.0,
            'params': None,
            'halflife': 0.0,
            'koff': 0.0,
            'status': 'fitting not performed',
            'start': fit_start,
            'end': fit_end,
            'message': 'fitting not performed',
            }
        return result

    fit_start = start
    if end == 0:
        fit_end = len(count_list)
    else:
        fit_end = end

    logger.info("Fitting from {0} to {1}.".format(fit_start, fit_end))
    time_list = time_list[fit_start:fit_end]
    count_list = count_list[fit_start:fit_end]

    times = np.array(time_list)
    counts = np.array(count_list)

    def one_phase_decay (params):
        a, b = params
        return np.sum(((a * np.exp(- b * times)) - counts) * ((a * np.exp(- b * times)) - counts))

    max_index = np.argmax(count_list)
    init_decay = float(count_list[max_index])
    init_params = [init_decay, 0.0]
    result_func = lambda x: params[0] * np.exp (- params[1] * x)

    opt = optimize.minimize(one_phase_decay, init_params, method = method)
    params = opt['x']
    result = {'func': result_func,
              'params': params,
              'halflife': np.log(2) / params[1],
              'koff': params[1],
              'status': opt['status'],
              'start': fit_start,
              'end': fit_end,
              'message': opt['message'],
    }
    return result

def regression (spot_table, time_scale = 1.0):
    if len(spot_table) == 0:
        return life_table(None)

    # regression
    work_table = spot_table.copy()
    start_plane = np.min(work_table.plane)
    if start_plane > 0:
        logger.info("Setting plane {0} as the beginning".format(start_plane))
        work_table.plane = work_table.plane - start_plane

    # spots to be counted
    index_set = set(work_table[work_table.plane == 0].total_index.tolist())
    logger.info("Regression set: {0}".format(index_set))

    # regression
    output_counts = []
    for index in range(work_table.plane.max() + 1):
        spot_count = len(work_table[(work_table.total_index.isin(index_set)) & (work_table.plane == index)])
        output_counts.append(spot_count)

    return life_table(output_counts, time_scale = time_scale)

def lifetime (spot_table, time_scale = 1.0, start_plane = 0):
    if len(spot_table) == 0:
        return life_table(None)

    # add lifetime
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # drop plane starting from the time-lapse image
    index_set = set(work_table[work_table.plane <= start_plane].total_index.tolist())
    logger.info("Dropping spots that start from plane {0}: {1}".format(start_plane, index_set))
    work_table = work_table[work_table.total_index.isin(index_set) == False]

    # prepare data (life_count starts from 0)
    work_table = work_table.drop_duplicates(subset = 'total_index', keep = 'last').reset_index(drop = True)
    life_max = work_table.life_count.max() + 1
    output_counts = [len(work_table[work_table.life_count == i]) for i in range(life_max)]

    return life_table(output_counts, time_scale = time_scale)

def cumulative (spot_table, time_scale = 1.0, start_plane = 0):
    if len(spot_table) == 0:
        return life_table(None)

    # add lifetime columns
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # drop plane starting from the time-lapse image
    index_set = set(work_table[work_table.plane <= start_plane].total_index.tolist())
    logger.info("Dropping spots that start from plane {0}: {1}".format(start_plane, index_set))
    work_table = work_table[work_table.total_index.isin(index_set) == False]

    # prepare data (life_count starts from 0)
    work_table = work_table.drop_duplicates(subset = 'total_index', keep = 'last').reset_index(drop = True)
    life_max = work_table.life_count.max() + 1
    output_counts = [len(work_table[work_table.life_count >= i]) for i in range(life_max)]

    return life_table(output_counts, time_scale = time_scale)

def new_bindings (spot_table, time_scale = 1.0):
    if len(spot_table) == 0:
        return binding_table(None, None)

    # add lifetime columns
    work_table = spot_table.copy()
    work_table['life_count'] = life_count(work_table)

    # prepare data (life_count starts from 0)
    agg_table = {'plane': np.min, 'life_count': np.max}
    work_table = work_table.groupby('total_index').agg(agg_table)
    
    plane_list = work_table['plane'].to_list()
    maxcount_list = work_table['life_count'].to_list()

    return binding_table(plane_list, maxcount_list, time_scale)
