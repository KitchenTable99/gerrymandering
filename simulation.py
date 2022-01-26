# the code for the Pixel and PixelMap classes
import argparse
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm as progress

from districts import District
from scorers import WeightValues, WeightDict


STATE_ABBREV = {'oh', 'ms', 'ny', 'ky', 'or', 'nv', 'wi', 'md', 'in', 'ct', 'ks', 'nd', 'sc', 'tn', 'ca', 'va', 'me',
                'sd', 'nm', 'la', 'dc', 'ok', 'mi', 'ri', 'ga', 'mn', 'ne', 'al', 'nh', 'mt', 'wv', 'fl', 'hi', 'ia',
                'pa', 'ar', 'nj', 'az', 'ma', 'il', 'nc', 'mo', 'ut', 'wa', 'ak', 'de', 'id', 'tx', 'co', 'vt', 'wy',
                'all'}


def border_generator(d: Dict[int, int], st_weight: int) -> Generator[int, None, None]:
    """This generator will create a list from a frequency dictionary, shuffle the list, then return the indices one
       by one.

       :param d: A dictionary with square_nums as keys and the number of new borders as values
       :param st_weight: The exponent to which the number of borders should be raised to get the frequency

       :return yields an integer which represents a square_num
    """
    # create frequency list
    freq_list = []
    for key, value in d.items():
        exp_value = value ** st_weight
        for _ in range(exp_value):
            freq_list.append(key)

    # yield random values in the list one by one
    np.random.shuffle(freq_list)
    for counter in range(len(freq_list) - 1):
        yield freq_list[counter]


@dataclass
class GerrymanderingSimulation:
    map: gpd.GeoDataFrame
    num_districts: int
    borders: Dict[int, int] = field(default_factory=dict)
    districts: List[District] = field(default_factory=list)

    logging: bool = False

    weight_dict: Optional[WeightDict] = None
    weights: Optional[WeightValues] = None
    score: float = -1.
    desired_results: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))

    def __post_init__(self):
        pixels = self.map.index.to_numpy()
        rows = self.map['row_num'].to_numpy()
        if self.logging:
            with open('log_changes.csv', 'w') as fp:
                fp.write('square_num,switch_district\n')
        self.pixel_to_row = {p: r for p, r in zip(pixels, rows)}
        self.weight_dict = WeightDict(WeightValues(3, np.array((10_000_000, 50, 1)), .5),
                                      WeightValues(1, np.array((30, -30, 30)), .75),
                                      WeightValues(3, np.array((3, 5, 5_000)), .5))

    def set_centering_weights(self) -> None:
        self.weights = self.weight_dict.centering

    def set_exploring_weights(self) -> None:
        self.weights = self.weight_dict.exploring

    def set_electioneering_weights(self) -> None:
        self.weights = self.weight_dict.electioneering

    def set_desired_results(self, results: np.ndarray) -> None:
        self.desired_results = results

    def tessellate(self) -> None:
        """This function creates a Vornoi tesselation based on the internal pixel map."""
        # convert points into an array to use in cdist
        point_array = self.map['np_geometry'].to_numpy()
        point_array = np.vstack(point_array)

        # randomly select self.num_districts unique Pixels in the map
        index = np.random.choice(point_array.shape[0], self.num_districts, replace=False)
        centroids = point_array[index]

        # find the closest centroid
        distances = distance.cdist(point_array, centroids)
        assignments = np.argmin(distances, axis=1)
        # assign each pixel to the appropriate class
        self.map['district'] = assignments

        if self.logging:
            self.map.to_pickle('log.pickle')

    def initialize_districts(self, verbose: bool = False) -> None:
        """This function initializes the districts by tessellating the map, finding the borders,
           creating district objects, and calculating starting properties of those districts."""
        # assign each pixel to a district
        if verbose:
            print('Tessellating...')
        self.tessellate()

        # find the borders using a BFS
        if verbose:
            print('Finding neighbors...')
            pbar = progress(total=len(self.map))

        # get all the stuff out of the DataFrame
        neighbors_arr = self.map['neighbors'].to_numpy()
        district_arr = self.map['district'].to_numpy()
        squares = self.map.index.to_numpy()
        borders = {}
        for row_num, square_num in enumerate(squares):
            if verbose:
                pbar.update(n=1)
            neighbors = neighbors_arr[row_num]
            focus_class = district_arr[row_num]

            neighbor_classes = [district_arr[self.pixel_to_row[neighbor]] for neighbor in neighbors]
            different_classes = [n_class != focus_class for n_class in neighbor_classes]
            borders[square_num] = sum(different_classes)

        if verbose:
            pbar.close()

        self.borders = borders

        # create the border objects
        if verbose:
            print('Creating districts...')
        self.districts = [District.from_df(self.map[self.map['district'] == district])
                          for district in range(self.num_districts)]

        # find the distance from each pixel to its population center
        if verbose:
            print('Populating district distances...')
        self.map.apply(lambda row: self.districts[row['district']].add_deviation(row['np_geometry']), axis=1)

        # score the map
        if verbose:
            print('Scoring...')
        self.weights = WeightValues(0, np.array((999999, 999999, 999999)), .5)
        self.score = self.evaluate(starting=True)

    def evaluate(self, districts: Optional[List[District]] = None, starting: bool = False) -> float:
        """This function scores a list of districts. It could be a set of actual districts or a hypothetical set when
           switching pixels. If no list is passed, this method will evaluate the internal districts.

           :param starting: Is this the first time the map is being scored?
           :param districts: A list of district objects to score

           :return the score represented by a float
        """
        # make sure that there are desired results before scoring
        if np.sum(self.desired_results) == 0:
            raise Exception('You need to specify election results before running the simulation')
        # see if method should score the internal districts
        if not districts:
            districts = self.districts

        scoring = np.zeros(3)
        # spread of population
        pop_arr = np.array([district.population for district in districts])
        pop_arr -= np.mean(pop_arr)
        scoring[0] = np.sum(np.power(pop_arr, 4))

        # spread of population from center
        scoring[1] = sum(district.deviation for district in districts) ** 4

        # calculate deviation from election results
        sorted_results = sorted(district.election_result for district in districts)
        least_squares = np.sum(np.square(np.diff(self.desired_results - sorted_results)))
        scoring[2] = least_squares

        return self.weights.scoring_weights.T @ scoring

    def district_breaks(self, first: int, second: int) -> bool:
        """This function determines if giving the first pixel the class of the second will break any district.

           :param first: the square_num of the first pixel
           :param second: the square_num of the second pixel

           :returns whether or not the switch breaks a district
        """
        # grab all the neighbors that might be broken
        check_neighbors = [neighbor for neighbor in self.map.at[first, 'neighbors'] if neighbor != second]
        class_to_neighbor_dict = {}
        for neighbor in check_neighbors:
            neighbor_class = self.map.at[neighbor, 'district']
            if neighbor_class not in class_to_neighbor_dict:
                class_to_neighbor_dict[neighbor_class] = [neighbor]
            else:
                class_to_neighbor_dict.get(neighbor_class).append(neighbor)

        # make sure all pairs can reach each other
        neighbors = self.map['neighbors'].to_numpy()
        districts = self.map['district'].to_numpy()
        for class_num, neighbor_list in class_to_neighbor_dict.items():
            # TODO: perform multi-stage BFS so that we don't search the huge chunk to completion

            # perform a BFS
            start = neighbor_list.pop()
            visited = [start]
            queue = [start]
            while queue:
                focus = queue.pop(0)
                if focus in neighbor_list:
                    neighbor_list.remove(focus)
                    if not neighbor_list:
                        break

                focus_neighbors = neighbors[self.pixel_to_row[focus]]
                for focus_neighbor in focus_neighbors:
                    # only visit the node if the node hasn't been visited, is in the correct class, and isn't the
                    # origin node
                    if (
                            focus_neighbor not in visited and
                            districts[self.pixel_to_row[focus_neighbor]] == class_num and
                            focus_neighbor != first
                    ):
                        visited.append(focus_neighbor)
                        queue.append(focus_neighbor)

            if neighbor_list:
                return True

        return False

    def eliminate_district(self,  remove_class: int) -> bool:
        return len(self.districts[remove_class].pixel_rows) <= 1

    def pick_swap_pair(self) -> Tuple[int, int]:
        """This function picks the pixel to switch.

           :returns the indices of the first and second pixels to switch
        """
        # randomly select the first pixel
        random_borders = border_generator(self.borders, self.weights.statistical_surface_tension)
        while True:
            # get a border pixel that we haven't tried yet
            first_pixel = next(random_borders)

            # find a neighbor with a different class
            first_class = self.map.at[first_pixel, 'district']
            neighbors = self.map.at[first_pixel, 'neighbors']
            random.shuffle(neighbors)

            for neighbor in neighbors:
                second_class = self.map.at[neighbor, 'district']
                if second_class == first_class or \
                        self.district_breaks(first_pixel, neighbor) or \
                        self.eliminate_district(first_class):
                    continue
                second_pixel = neighbor
                break
            else:
                continue
            return first_pixel, second_pixel

    def swap_pixels(self, first: int, second: int) -> None:
        # any reference to districts will be f_district or s_district for first or second district
        # make copies of original districts
        f_class = self.map.at[first, 'district']
        s_class = self.map.at[second, 'district']
        f_district = self.districts[f_class]
        s_district = self.districts[s_class]
        old_f_deviation = f_district.deviation
        old_s_deviation = s_district.deviation

        # remove pixel
        first_data = self.map.loc[first]
        f_district.remove_pixel(first_data)
        # add pixel
        s_district.add_pixel(first_data)

        # reset districts and calculate deviations
        f_district.reset_deviation()
        s_district.reset_deviation()
        # these two lines basically perform a pd.apply but vectorized for performance
        geom = self.map['np_geometry'].to_numpy()
        districts = self.map['district'].to_numpy()
        first_geom = geom[districts == f_class]
        second_geom = geom[districts == s_class]
        v = np.vectorize(lambda d, g: d.add_deviation(g))
        v(f_district, first_geom)
        v(s_district, second_geom)

        # re-score the map
        new_score = self.evaluate()

        # if the map is worse and fails the vibe check, return
        if new_score > self.score and random.random() > self.weights.keep_bad_maps:
            f_district.add_pixel(first_data)
            s_district.remove_pixel(first_data)
            f_district.deviation = old_f_deviation
            s_district.deviation = old_s_deviation
            return

        # this change was accepted--log the change
        if self.logging:
            to_append = str(first)
            to_append += ','
            to_append += str(s_class)
            to_append += '\n'
            with open('log_changes.csv', 'a') as fp:
                fp.write(to_append)

        # update the score, district objects, pixel classification
        self.score = new_score
        self.map.at[first, 'district'] = self.map.at[second, 'district']

        # update border status of the swapped pixel and all its neighbors
        update_borders = {first}
        for pixel in self.map.at[first, 'neighbors']:
            update_borders.add(pixel)
        for pixel in update_borders:
            focus_neighbors = self.map.at[pixel, 'neighbors']
            f_neighbor_class = [self.map.at[f_neighbor, 'district'] for f_neighbor in focus_neighbors]
            pixel_class = self.map.at[pixel, 'district']
            different_classes = [n_class != pixel_class for n_class in f_neighbor_class]
            self.borders[pixel] = sum(different_classes)

    def gerrymander(self, num_center_swaps: int, num_explore_swaps: int, num_electioneer_swaps: int) -> None:
        if np.sum(self.desired_results) == 0:
            raise Exception('You need to set the desired results')
        # three phases
        self.set_centering_weights()
        for _ in progress(range(num_center_swaps), desc='Centering'):
            first, second = self.pick_swap_pair()
            self.swap_pixels(first, second)

        self.set_exploring_weights()
        for _ in progress(range(num_explore_swaps), desc='Exploring'):
            first, second = self.pick_swap_pair()
            self.swap_pixels(first, second)

        self.set_electioneering_weights()
        for _ in progress(range(num_electioneer_swaps), desc='Electioneering'):
            first, second = self.pick_swap_pair()
            self.swap_pixels(first, second)

    def show_districts(self):
        """This function plots a choropleth of the internal GeoDataFrame with the district as the color var."""
        self.map.plot(column='district')
        plt.show()

    def plot_column(self, column: str, cmap: str = 'viridis') -> None:
        """This function will plot some column. Useful for visualizing lots of stuff.

        :param column: the column to map the color to.
        :param cmap: the colormap to use
        """
        self.map.plot(column=column, cmap=cmap)
        plt.show()


def driver(state: str, num_districts: int, testing: bool, verbose: bool, logging: bool, show_after: bool,
           centering: int = 120_000, exploring: int = 50_000, electioneering: int = 120_000) -> None:
    pickle_path = f'./data/{"test_maps" if testing else "maps"}/{state}.pickle'
    with open(pickle_path, 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)

    sim = GerrymanderingSimulation(pixel_map, num_districts, logging=logging)
    sim.set_desired_results(np.repeat(.6, num_districts))
    sim.initialize_districts(verbose=verbose)
    # TODO: update this with tuned hyper-parameters
    sim.gerrymander(centering, exploring, electioneering)

    sim.map.to_pickle('simulation_out.pickle')
    print(f'There were {len(pd.unique(sim.map["district"]))} districts')

    if show_after:
        sim.show_districts()


def get_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This program runs a simulation for a passed state. All can be passed '
                                                 'to run every state.')
    parser.add_argument('state', type=str, choices=STATE_ABBREV, help='The state to simulate. If all is '
                                                                      'passed, all 50 states will be simulated.')
    parser.add_argument('num_districts', type=int, help='The number of districts to carve the map into.')
    parser.add_argument('--testing', '-t', action='store_true', help='Simulate testing maps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print out debug information')
    parser.add_argument('--logging', action='store_true', help='Log individual pixel swaps')
    parser.add_argument('--profile', action='store_true', help='Profile the simulation')
    parser.add_argument('--show', action='store_true', help='Show the map at the end of the simulation')
    parser.add_argument('--specify', type=int, nargs=3, help='The number of iterations for which to gerrymander')

    return parser.parse_args()


def main():
    cmd_args = get_cmd_args()
    if cmd_args.state == 'all':
        raise NotImplementedError

    if cmd_args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            explore, center, electioneer = cmd_args.specify or (120_000, 50_000, 120_000)
            driver(cmd_args.state, cmd_args.num_districts, cmd_args.testing, cmd_args.verbose, cmd_args.logging,
                   cmd_args.show, explore, center, electioneer)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename='profile.prof')
    else:
        explore, center, electioneer = cmd_args.specify or (120_000, 50_000, 120_000)
        driver(cmd_args.state, cmd_args.num_districts, cmd_args.testing, cmd_args.verbose, cmd_args.logging,
               cmd_args.show, explore, center, electioneer)


if __name__ == '__main__':
    main()
