#!/usr/bin/env python

import sys, argparse, json, graphviz
import pandas as pd
from pathlib import Path
from mmtools import mmtiff, trackj, particles

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
output_filename = args.output_file
if output_filename is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)

# read JSON or TSV or TrackJ CSV file
suffix = Path(input_filename).suffix.lower()
if suffix == '.json':
    with open(input_filename, 'r') as f:
        records = json.load(f)        
        spot_table = pd.DataFrame(particles.parse_tree(records['spot_list']), dtype = object)
elif suffix == ".txt":
    spot_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
elif suffix == ".csv":
    spot_table = trackj.read_spots(input_filename)
else:
    raise Exception("Unknown file format.")
    
total_records = len(spot_table.total_index)
total_tracks = len(spot_table.total_index.unique())
print("{0}: {1} records and {2} tracks.".format(input_filename, total_records, total_tracks))

graph = graphviz.Graph(engine = 'dot', format = Path(output_filename).suffix[1:])

graph.node("Start")
graph.attr("node", shape = "rectangle")
for index in spot_table.total_index.unique():
    track_table = spot_table[spot_table.total_index == index]
    node = "Start"
    for track_index in range(len(track_table)):
        current = "Spot_{0}".format(track_table.index[track_index])
        graph.edge(node, current)
        node = current

graph.render(Path(output_filename).stem)