# the code for the Pixel and PixelMap classes
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm as progress

from districts import District
from scorers import WeightValues, WeightDict


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

    weight_dict: Optional[WeightDict] = None
    weights: Optional[WeightValues] = None
    score: float = -1.
    desired_results: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))

    def __post_init__(self):
        self.weight_dict = WeightDict(WeightValues(3, np.diag((10, 20, 3)), .5),
                                      WeightValues(1, np.diag((1, 1, 1)), .75),
                                      WeightValues(3, np.diag((1, 2, 30)), .5))

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

    def initialize_districts(self) -> None:
        """This function initializes the districts by tessellating the map, finding the borders,
           creating district objects, and calculating starting properties of those districts."""
        # assign each pixel to a district
        self.tessellate()

        # find the borders using a BFS
        borders = {}
        visited = [self.map.index.values[0]]
        queue = [self.map.index.values[0]]
        while queue:
            focus = queue.pop(0)
            neighbors = self.map.at[focus, 'neighbors']
            focus_class = self.map.at[focus, 'district']

            neighbor_classes = []
            for neighbor in neighbors:
                neighbor_classes.append(self.map.at[neighbor, 'district'])
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)

            different_classes = [n_class != focus_class for n_class in neighbor_classes]
            borders[focus] = sum(different_classes)

        self.borders = borders

        # create the border objects
        self.districts = [District.from_df(self.map[self.map['district'] == district])
                          for district in range(self.num_districts)]

        # find the distance from each pixel to its population center
        self.map.apply(lambda row: self.districts[row['district']].add_deviation(row['np_geometry']), axis=1)

        # score the map
        self.weights = WeightValues(0, np.diag((100, 100, 100)), .5)
        self.score = self.evaluate()

    def evaluate(self, districts: Optional[List[District]] = None) -> float:
        """This function scores a list of districts. It could be a set of actual districts or a hypothetical set when
           switching pixels. If no list is passed, this method will evaluate the internal districts.

           :param districts: A list of district objects to score

           :return the score represented by a float
        """
        # make sure that there are desired results before scoring
        # TODO: put this in a better place
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
        scoring[1] = sum(district.deviation for district in districts)

        # calculate deviation from election results
        # TODO: actually fit the results to some curve
        scoring[2] = 1

        return np.sum(self.weights.scoring_weights @ scoring.T)

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
        for class_num, neighbor_list in class_to_neighbor_dict.items():
            # only bother if there are at least two neighbors
            # TODO: perform multi-stage BFS so that we don't search the huge chunk to completion
            if len(neighbor_list) < 2:
                continue

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

                focus_neighbors = self.map.at[focus, 'neighbors']
                for focus_neighbor in focus_neighbors:
                    # only visit the node if the node hasn't been visited, is in the correct class, and isn't the
                    # origin node
                    if (
                            focus_neighbor not in visited and
                            self.map.at[focus_neighbor, 'district'] == class_num and
                            focus_neighbor != first
                    ):
                        visited.append(focus_neighbor)
                        queue.append(focus_neighbor)

            if neighbor_list:
                return True

        return False

    def eliminate_district(self, remove_pixel: int) -> bool:
        """If the passed pixel is the last in its class, this function will return True

           :param remove_pixel: the square_num of the pixel to test

           :return whether or not the pixel is the last in the district
        """
        remove_pop = self.map.at[remove_pixel, 'population']
        remove_class = self.map.at[remove_pixel, 'district']

        return self.districts[remove_class].population - remove_pop <= 0

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
                        self.eliminate_district(first_pixel):
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
        f_district_copy = deepcopy(self.districts[f_class])
        s_district_copy = deepcopy(self.districts[s_class])

        # remove pixel
        first_data = self.map.loc[first]
        f_district_copy.remove_pixel(first_data)
        # add pixel
        s_district_copy.add_pixel(first_data)

        # recalculate deviations
        for district in self.districts:
            district.reset_deviation()
        # these two lines basically perform a pd.apply but vectorized for performance
        v = np.vectorize(lambda d, g: self.districts[d].add_deviation(g))
        v(self.map.district, self.map.np_geometry)

        # re-score the map
        to_score = deepcopy(self.districts)
        to_score[f_class] = f_district_copy
        to_score[s_class] = s_district_copy
        new_score = self.evaluate(to_score)

        # if the map is worse and fails the vibe check, return
        if new_score > self.score and random.random() > self.weights.keep_bad_maps:
            return

        # update the score, district objects, pixel classification
        self.score = new_score
        self.districts = to_score
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


def test():
    with open('test_map.pickle', 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)

    sim = GerrymanderingSimulation(pixel_map, 13)

    sim.set_desired_results(np.array([1, 2, 3]))
    sim.initialize_districts()
    sim.gerrymander(10_000, 1000, 10_000)

    sim.show_districts()


def main():
    with open('test_map.pickle', 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)

    sim = GerrymanderingSimulation(pixel_map, 13)
    sim.set_desired_results(np.repeat(.6, 13))

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        sim.initialize_districts()
        sim.gerrymander(100_000, 1000, 10_000)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profile.prof')


if __name__ == '__main__':
    # main()
    test()
