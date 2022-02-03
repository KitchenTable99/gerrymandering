# This file contains diagnostics for PixelMaps
import argparse
import pickle

import matplotlib.pyplot as plt


# TODO: figure out diagnostics for the completed map
STATE_ABBREV = {'oh', 'ms', 'ny', 'ky', 'or', 'nv', 'wi', 'md', 'in', 'ct', 'ks', 'nd', 'sc', 'tn', 'ca', 'va', 'me',
                'sd', 'nm', 'la', 'dc', 'ok', 'mi', 'ri', 'ga', 'mn', 'ne', 'al', 'nh', 'mt', 'wv', 'fl', 'hi', 'ia',
                'pa', 'ar', 'nj', 'az', 'ma', 'il', 'nc', 'mo', 'ut', 'wa', 'ak', 'de', 'id', 'tx', 'co', 'vt', 'wy'}


def driver(state: str, year: int, testing: bool):
    pickle_path = f'./data/{"test_maps" if testing else "maps"}/{year}/{state}.pickle'
    with open(pickle_path, 'rb') as fp:
        pixel_map = pickle.load(fp)

    fig, ax = plt.subplots(2, 2)

    pixel_map.loc[pixel_map.population == 0].plot(ax=ax[0, 0])
    ax[0, 0].title.set_text('Zero Population')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    pixel_map.plot(column='population', ax=ax[0, 1])
    ax[0, 1].title.set_text('Population')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    pixel_map.plot(column='red_votes', ax=ax[1, 0])
    ax[1, 0].title.set_text('Republican Votes')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    pixel_map.plot(column='blue_votes', ax=ax[1, 1])
    ax[1, 1].title.set_text('Democratic Votes')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])

    plt.tight_layout()
    plt.show()
    # TODO: implement loss statistics


def get_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This program investigates a state\'s map. If testing is passed, the '
                                                 'testing map will be investigated.')
    parser.add_argument('state', type=str, choices=STATE_ABBREV, help='The state for which to check diagnostics.')
    parser.add_argument('year', type=int, choices={2016, 2020}, help='The year for which to check diagnostics.')
    parser.add_argument('--testing', '-t', action='store_true', help='Diagnose testing maps')

    return parser.parse_args()


def main():
    cmd_args = get_cmd_args()
    print(f'Reading {cmd_args.state.upper()}...')
    driver(cmd_args.state, cmd_args.year, cmd_args.testing)


if __name__ == '__main__':
    main()
