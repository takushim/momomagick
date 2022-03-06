#!/usr/bin/env python

import argparse, json, graphviz
from pathlib import Path
from mmtools import stack, particles

# default values
input_filename = None
output_filename = None
output_suffix = "_tree.svg"

# parse arguments
parser = argparse.ArgumentParser(description='Draw a tree diagram of particles using GraphViz', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output filenames ([basename]_{0} if not specified)'.format(output_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='input JSON file of tracking data.')

args = parser.parse_args()

# set arguments
input_filename = args.input_file
if args.output_file is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# read JSON or TSV or TrackJ CSV file
with open(input_filename, 'r') as f:
    spot_list = json.load(f)['spot_list']
    spot_list = [spot for spot in spot_list if spot.get('delete', False) == False]

# draw graph
graph = graphviz.Graph(engine = 'dot', format = Path(output_filename).suffix[1:])
graph.attr("node", shape = "rectangle")

def add_child_edges (spot, parent):
    name = "Spot_{0}".format(spot['index'])
    label = "Spot_{0} (t = {1})\nx{2:.2f} y{3:.2f} z{4}".format(spot['index'], spot['time'], spot['x'], spot['y'], spot['z'])
    graph.node(name = name, label = label)
    if parent is None:
        graph.node(name = name, label = label)
    else:
        graph.node(name = name, label = label)
        graph.edge("Spot_{0}".format(spot['index']), "Spot_{0}".format(parent['index']))

    child_list = particles.find_children(spot, spot_list)
    for child in child_list:
        add_child_edges(child, spot)

root_list = [spot for spot in spot_list if spot['parent'] is None]
for root in root_list:
    add_child_edges(root, None)

graph.render(Path(output_filename).stem)