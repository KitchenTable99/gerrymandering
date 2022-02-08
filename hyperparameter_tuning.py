# this script will provide a grid of hyperparameters
import concurrent.futures
import pickle
import multiprocessing
import random
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

from simulation import GerrymanderingSimulation
import argparse


STATE_ABBREV = {'oh', 'ms', 'ny', 'ky', 'or', 'nv', 'wi', 'md', 'in', 'ct', 'ks', 'nd', 'sc', 'tn', 'ca', 'va', 'me',
                'sd', 'nm', 'la', 'dc', 'ok', 'mi', 'ri', 'ga', 'mn', 'ne', 'al', 'nh', 'mt', 'wv', 'fl', 'hi', 'ia',
                'pa', 'ar', 'nj', 'az', 'ma', 'il', 'nc', 'mo', 'ut', 'wa', 'ak', 'de', 'id', 'tx', 'co', 'vt', 'wy',
                'all'}


def get_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This program runs a simulation for a passed state. All can be passed '
                                                 'to run every state.')
    parser.add_argument('state', type=str, choices=STATE_ABBREV, help='The state to simulate. If all is '
                                                                      'passed, all 50 states will be simulated.')
    parser.add_argument('num_districts', type=int, help='The number of districts to carve the map into.')
    parser.add_argument('year', type=int, choices={2016, 2020}, help='Which year to simulate')
    parser.add_argument('--testing', '-t', action='store_true', help='Simulate testing maps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print out debug information')
    parser.add_argument('--logging', '-l', type=str, choices={'swaps', 'score'}, help='Log either the swaps '
                                                                                      'or score.')
    parser.add_argument('--log_name', type=str, default='log', help='The name of the log file. Defaults to log')
    parser.add_argument('--profile', action='store_true', help='Profile the simulation')
    parser.add_argument('--show', action='store_true', help='Show the map at the end of the simulation')
    parser.add_argument('--specify', type=int, nargs=3, help='The number of iterations for which to gerrymander')

    return parser.parse_args()


def rolling_min(minimize: pd.Series) -> pd.Series:
    mins = np.zeros(len(minimize))
    current_min = minimize[0]
    for idx, num in enumerate(minimize):
        if num < current_min:
            current_min = num
        mins[idx] = current_min

    return pd.Series(mins)


def sim_driver(pixel_map: gpd.GeoDataFrame, num_districts: int, verbose: bool, log_name: str, centering: int,
               exploring: int, electioneering) -> None:
    # TODO: the testing needs to already be done here
    sim = GerrymanderingSimulation(pixel_map, num_districts)
    sim.set_logging('score', log_name)
    sim.set_desired_results(np.repeat(.6, num_districts))
    sim.initialize_districts(verbose=verbose)
    try:
        sim.gerrymander(centering, exploring, electioneering)
    except KeyboardInterrupt:
        print(f'There were {len(pd.unique(sim.map["district"]))} districts')
        sim.show_districts()


def driver(cmd_args: argparse.Namespace):
    pickle_path = f'./data/{"test_maps" if cmd_args.testing else "maps"}/{cmd_args.year}/{cmd_args.state}.pickle'
    with open(pickle_path, 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)

    big_df = pd.DataFrame()
    success_count = 0
    while success_count < 5:
        try:
            sim_driver(pixel_map, cmd_args.num_districts, cmd_args.verbose, 'hyper_tuning', 1_000_000, 0, 0)
            success_count += 1
        except ZeroDivisionError:
            continue
        df = pd.read_csv('hyper_tuning.csv')
        big_df[str(success_count)] = df.score[1:]

    big_df['mean'] = big_df.mean(axis=1)
    big_df.plot()
    big_df.to_pickle('averaged.pickle')
    plt.show()


def main():
    cmd_args = get_cmd_args()
    driver(cmd_args)


if __name__ == '__main__':
    main()
