# the code for the Pixel and PixelMap classes
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil, dist
from typing import List, Dict, Tuple, Generator, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely as shapely
from scipy.spatial import distance
from shapely.geometry import Polygon
from scorers import *


def pixel_to_center(pixel_centroid: shapely.geometry.Point, district: 'District') -> float:
    # find centers
    pixel_center = (pixel_centroid.x, pixel_centroid.y)
    district_center = district.get_center()

    # calculate distance
    return dist(pixel_center, district_center)


def initialize_pixel_dist(row: pd.Series, districts: List['District']) -> None:
    district = districts[row['district']]
    pixel_centroid = row['geometry'].centroid

    pix_dist = pixel_to_center(pixel_centroid, district)
    district.add_pixel_deviation(row['square_num'], pix_dist)
    district.add_pixel_centroid(row['square_num'], pixel_centroid)


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


def shapely_to_array(points: List[shapely.geometry.Point]) -> np.ndarray:
    """This function converts a list of shapely points into an array of points to be used in cdist.

        :param points. A list of points from the shapely library

        :returns an array of points
    """
    point_tuples = [(point.x, point.y) for point in points]

    return np.array([*point_tuples])


@dataclass
class District:

    population: float
    population_center: np.ndarray
    red_votes: float
    blue_votes: float

    pixel_deviations: Dict[int,  float] = field(default_factory=dict)
    pixel_centroids: Dict[int, shapely.geometry.Point] = field(default_factory=dict)

    @property
    def election_result(self) -> float:
        return self.red_votes / self.blue_votes

    @property
    def deviation(self) -> float:
        return sum(self.pixel_deviations.values())

    def get_center(self) -> np.ndarray:
        return self.population_center

    def add_pixel_deviation(self, pix_num: int, pix_dist: float) -> None:
        """This function adds an entry into the internal pixel deviations dictionary. If an entry is already present,
           nothing further happens.

           :param pix_num: the number of the pixel. the GeoDataFrame refers to this as square_num
           :param pix_dist: the distance the center of that pixel is from the center of the district
        """
        if self.pixel_deviations.get(pix_num):
            return

        self.pixel_deviations[pix_num] = pix_dist

    def add_pixel_centroid(self, pix_num: int, pix_centroid: shapely.geometry.Point) -> None:
        """This function adds an entry into the internal pixel deviations dictionary. If an entry is already present,
           nothing further happens.

           :param pix_num: the number of the pixel. the GeoDataFrame refers to this as square_num
           :param pix_centroid: the centroid of the pixel to add
        """
        if self.pixel_centroids.get(pix_num):
            return

        self.pixel_centroids[pix_num] = pix_centroid

    def add_pixel(self, pix_pop: float, pix_red: float, pix_blue: float, pix_num: int, pix_center: shapely.geometry.Point) -> None:
        # update the basic sum quantities
        self.population += pix_pop
        self.red_votes += pix_red
        self.blue_votes += pix_blue

        # edit center
        # TODO: actually edit the center

        # update the old pixels with the new center
        for pix, centroid in self.pixel_centroids.items():
            new_dist = pixel_to_center(centroid, self)
            self.pixel_deviations[pix] = new_dist
        # add in the new pixel to the new center


@dataclass
class GerrymanderingSimulation:
    map: gpd.GeoDataFrame
    num_districts: int
    borders: Dict[int, int] = field(default_factory=dict)
    districts: List[District] = field(default_factory=list)

    score: float = -1.
    weight_dict: WeightDict = field(default_factory=WeightDict)
    desired_results: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))

    def __post_init__(self):
        self.weight_dict = WeightDict(WeightValues(3, np.diag((10, 20, 3)), .5),
                                      WeightValues(1, np.diag((1, 1, 1)), .75),
                                      WeightValues(3, np.diag((1, 2, 30)), .5))

    def tessellate(self) -> None:
        """This function creates a Vornoi tesselation based on the internal pixel map."""
        # convert points into an array to use in cdist
        points = [poly.centroid for poly in self.map['geometry'].values]
        point_array = shapely_to_array(points)

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
        self.create_district_objs()

        # find the distance from each pixel to its population center
        self.map.apply(lambda row: initialize_pixel_dist(row, self.districts), axis=1)

        # score the map
        self.score = self.evaluate()

    def create_district_objs(self) -> None:
        """This function creates the district objects that are used to calculate score."""
        # rotate over each district
        district_objs = []
        for district in range(self.num_districts):
            # find simple sum metrics
            gdf = self.map[self.map['district'] == district]
            pop, red, blue = gdf[['population', 'red_votes', 'blue_votes']].sum()

            # find population center
            shapely_points = gdf['geometry'].centroid
            points = shapely_to_array(shapely_points)
            center = np.average(points, weights=gdf['population'].to_numpy(), axis=0)

            district_objs.append(District(pop, center, red, blue))

        self.districts = district_objs

    def evaluate(self, districts: Optional[List[District]] = None) -> float:
        """This function scores a list of districts. It could be a set of actual districts or a hypothetical set when
           switching pixels. If no list is passed, this method will evaluate the internal districts.

           :param districts: A list of district objects to score

           :return the score represented by a float
        """
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

        return self.weights @ scoring.T  # TODO: check that this returns a float

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

        return self.populations[remove_class] - remove_pop <= 0

    def pick_swap_pair(self) -> Tuple[int, int]:
        """This function picks the pixel to switch.

           :returns the indices of the first and second pixels to switch
        """
        # randomly select the first pixel
        random_borders = border_generator(self.borders, self.surface_tension_weight)
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
                        self.eliminate_district(first_pixel) or \
                        self.district_breaks(first_pixel, neighbor):
                    continue
                second_pixel = neighbor
                break
            else:
                continue
            return first_pixel, second_pixel

    def swap_pixels(self, first: int, second: int) -> bool:
        """This function tries to swap the passed pixels. Always swap if the map is better, randomly swap if map is
           worse. Returns whether or not the swap took place.

           :param first: the pixel to inherit the class of the second
           :param second: the pixel to give the class to first

           :returns whether or not a swap took place
        """
        # o_district will represent the original district
        # d_district will represent the destination district
        o_district = self.map.at[first, 'district']
        d_district = self.map.at[second, 'district']

        # keep a copy of the two districts that will change
        old_o_district = deepcopy(self.districts[o_district])
        old_d_district = deepcopy(self.districts[d_district])

        # swap district
        self.map.at[first, 'district'] = d_district

        # update metrics
        pop = self.map.at[first, 'population']
        red = self.map.at[first, 'red_votes']
        blue = self.map.at[first, 'blue_votes']

        # create new districts
        # population
        origin_new_pop = self.populations[o_district] - pop
        destination_new_pop = self.populations[d_district] + pop
        # centers
        # this is fairly easy to follow if you remove an arbitrary item from an average
        # back of the napkin math shows the formula
        # the relative unreadability is just because of how funky it is to type out math
        # TODO: fix this formula to use weighted count (population)
        c = self.populations[o_district]
        n = self.counts[o_district]
        pixel_point = self.map.at[first, 'geometry'].centroid
        pixel_arr = np.array([pixel_point.x, pixel_point.y])

        origin_new_center = ((c*n) - pixel_arr) / (n - 1)

        d = self.populations[d_district]
        n = self.counts[d_district]

        destination_new_center = ((d*n) + pixel_arr) / (n + 1)
        # results
        origin_new_red = self.red_totals[o_district] - red
        origin_new_blue = self.blue_totals[o_district] - blue
        destination_new_red = self.red_totals[d_district] + red
        destination_new_blue = self.red_totals[d_district] + blue

        # score itself
        # duplicate the arrays
        n_populations = self.populations.copy()
        n_populations[o_district] = origin_new_pop
        n_populations[d_district] = destination_new_pop

        n_centers = self.centers.copy()
        n_centers[o_district] = origin_new_center
        n_centers[d_district] = destination_new_center

        n_red = self.red_totals.copy()
        n_red[o_district] = origin_new_red
        n_red[d_district] = destination_new_red

        n_blue = self.blue_totals.copy()
        n_blue[o_district] = origin_new_blue
        n_blue[d_district] = destination_new_blue

        new_score = self.evaluate(n_populations, n_centers, n_red, n_blue)

        if new_score > self.score and random.random() < self.keep_bad_maps:
            self.map.at[first, 'class'] = o_district
            return False

        # update the internal fields
        self.counts[o_district] -= 1
        self.counts[d_district] += 1

        self.populations = n_populations
        self.centers = n_centers
        self.red_totals = n_red
        self.blue_totals = n_blue

        # check swapped pixel and its neighbors for border state
        update_borders = {first}
        for neighbor in self.map.at[first, 'neighbors']:
            update_borders.add(neighbor)

        for pixel in update_borders:
            focus_neighbors = self.map.at[pixel, 'neighbors']
            f_neighbor_class = [self.map.at[f_neighbor, 'district'] for f_neighbor in focus_neighbors]
            pixel_class = self.map.at[pixel, 'district']
            different_classes = [n_class != pixel_class for n_class in f_neighbor_class]
            self.borders[pixel] = sum(different_classes)

        return True

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
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    pixel_map.weights = np.array([1, 2, 3])
    pixel_map.desired_results = np.array([1, 2, 3])
    pixel_map.initialize_districts()

    fig, ax = plt.subplots()
    pixel_map.map.plot(column='district', ax=ax)
    ax.axis('off')
    plt.show()

    # for _ in range(100):
    #     first, second = pixel_map.pick_swap_pair()
    #     pixel_map.swap_pixels(first, second)


def main():
    nc_votes = gpd.read_file('./nc_data/voters_shapefile/NC_G18.shp')
    nc_pop = gpd.read_file('./nc_data/population.geojson')
    nc_votes = nc_votes[['G18DStSEN', 'G18RStSEN', 'geometry']].rename(
        columns={'G18DStSEN': 'blue_votes', 'G18RStSEN': 'red_votes'})

    nc_pixel_map = PixelMap(nc_votes, nc_pop, 1, 13)
    # TODO: find a way to set these metrics
    nc_pixel_map.keep_bad_maps = .5
    nc_pixel_map.surface_tension_weight = 3
    nc_pixel_map.weights = np.array([1, 2, 3])
    nc_pixel_map.desired_results = np.array([1, 2, 3])
    nc_pixel_map.initialize_districts()

    with open('pix.pickle', 'wb') as fp:
        pickle.dump(nc_pixel_map, fp)

    # import cProfile
    # import pstats
    #
    # with cProfile.Profile() as pr:
    #     pass
    #
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename='profile.prof')


if __name__ == '__main__':
    # main()
    test()
