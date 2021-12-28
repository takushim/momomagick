#!/usr/bin/env python

class TreeParseException (Exception):
    def __init__ (self, message = "Unknown exception"):
        self.message = message
    def __str__ (self):
        return self.message

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

