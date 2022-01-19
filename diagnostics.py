# This file contains diagnostics for PixelMaps
import pickle
import sys

import matplotlib.pyplot as plt


# TODO: figure out diagnostics for the completed map


def usage_statement():
    """This prints the usage statement and exits."""
    print('\nUSAGE STATEMENT:')
    print(f'python3 {sys.argv[0]} [STATE NAME] [TESTING]')
    print('\n\nThis file takes in the name of a United State and whether or not to run the testing mode'
          'block level. The name of the state must be a lower-cased, two-letter abbreviation (e.g. nc, ca). '
          'It writes this data to a geojson file to the current working directory.')
    sys.exit()


def main(state: str, testing: bool):
    pickle_path = f'./data/{"test_maps" if testing else "maps"}/{state}.pickle'
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


if __name__ == '__main__':
    try:
        _, state, testing = sys.argv
        test = True if testing == 'test' or testing == 'testing' or testing == '-t' else False
    except Exception:
        usage_statement()

    main(state, test)
