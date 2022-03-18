#!/usr/bin/env python

import json
import pandas as pd
from numpyencoder import NumpyEncoder
from datetime import datetime

class TreeParseException (Exception):
    def __init__ (self, message = "Unknown exception"):
        self.message = message
    def __str__ (self):
        return self.message

def load_json (filename, keep_delete = False):
    with open(filename, 'r') as file:
        json_data = json.load(file)
        spot_list = json_data.get('spot_list', [])
        if keep_delete == False:
            json_data['spot_list'] = [spot for spot in spot_list if spot['delete'] == False]
    return json_data

def save_json (filename, json_data):
    with open(filename, 'w') as file:
        json.dump(json_data, file, ensure_ascii = False, indent = 4, sort_keys = False, \
                  separators = (',', ': '), cls = NumpyEncoder)

def load_spots (filename, keep_delete = False):
    with open(filename, 'r') as file:
        spot_list = json.load(file).get('spot_list', [])
        if keep_delete == False:
            spot_list = [spot for spot in spot_list if spot['delete'] == False]
    return spot_list

def list_to_table (spot_list):
    if spot_list is None or len(spot_list) == 0:
        dummy_spot = create_spot(index = 0, time = 0, channel = 0, x = 0.0, y = 0.0, z = 0, parent = None)
        spot_table = pd.DataFrame(parse_tree([dummy_spot]))
        spot_table = spot_table.drop(spot_table.index)
    else:
        spot_table = pd.DataFrame(parse_tree(spot_list))
    
    return spot_table

def parse_tree (spot_list):
    spot_list = [spot for spot in spot_list if spot.get('delete', False) == False]
    leaf_list = [spot for spot in spot_list if len(find_children(spot, spot_list)) == 0]

    output_list = []
    for index in range(len(leaf_list)):
        current = leaf_list[index]
        track_list = []
        while current is not None:
            current['track'] = index
            track_list.append(current)
            parent = [spot for spot in spot_list if spot['index'] == current['parent']]

            if len(parent) == 0:
                current = None
            elif len(parent) == 1:
                current = parent[0]
            else:
                raise TreeParseException('spot {0} has {1} parents.'.format(current['index'], len(parent)))

            if current in track_list:
                raise TreeParseException('Loop detected while tracking spot {0}'.format(current['index']))

        output_list.extend(reversed(track_list))

    return output_list

def find_children (spot, spot_list):
    return [x for x in spot_list if (x['parent'] == spot['index']) and (x['delete'] == False)]

def create_spot (index = None, time = None, channel = None, x = None, y = None, z = None, parent = None):
    spot = {'index': index, 'time': time, 'channel': channel, \
            'x': x, 'y': y, 'z': z, 'parent': parent, 'label': None, \
            'delete': False, \
            'create': datetime.now().astimezone().isoformat(), \
            'update': datetime.now().astimezone().isoformat()}
    return spot
