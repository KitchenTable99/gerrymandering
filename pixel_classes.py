# the code for the Pixel and PixelMap classes
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm as progress
from math import ceil
import geopandas as gpd
import pandas as pd
import shapely as shapely
import matplotlib.pyplot as plt
from scipy.spatial import distance
from shapely.geometry import Polygon
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Generator


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


def weighted_intersection(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, attr_str: str) -> gpd.GeoDataFrame:
    """This function takes two GeoDataFrames with disjoint geometry and puts some target attribute from the second into
        the first one. This function will overlay the two dataframes and if poly1 from gdf1 intersects poly2 from gdf2
        and the area of the intersectional polygon is 50% of gdf2, 50% of the target attribute from gdf2 will be assigned
        to poly1 in gdf1.

        :param gdf1. The dataframe to gain the target attribute
        :param gdf2. The dataframe from which to take the target attribute
        :param attr_str. The attribute to take. Must be a column name in gdf2.

        :returns the GeoDataFrame with the new attribute tacked on
    """
    # input checking
    assert attr_str in gdf2.columns, 'attr_str must be in gdf2.columns'

    # prepare GeoDataFrames
    gdf1 = gdf1.to_crs(3857)
    if 'square_num' not in gdf1.columns:
        gdf1.insert(0, 'square_num', range(len(gdf1)))

    gdf2 = gdf2.to_crs(3857)
    gdf2.insert(0, 'gdf2_num', range(len(gdf2)))
    if 'area' not in gdf2.columns:
        gdf2['area'] = gdf2.area

    # overlay the two data frames and calculate the proportionality of overlap
    inter = gpd.overlay(gdf1, gdf2, how='intersection')
    inter['inter_area'] = inter.area
    inter['gdf2_area'] = inter.apply(lambda r: gdf2.at[r['gdf2_num'], 'area'], axis=1)
    # this gets the area of the gdf2 polygon that the smaller inter polygon came from
    inter['prop ' + attr_str] = (inter['inter_area'] / inter['gdf2_area']) * inter[attr_str]
    inter = inter[['square_num', 'prop ' + attr_str]]

    # group all the smaller inter polygons by square and put them back in the original GeoDataFrame via sum aggregation
    groups = inter.groupby('square_num').sum()
    gdf1['attr_str'] = np.nan
    for row in groups.iterrows():
        square, attr = row
        attr = float(attr)
        gdf1.at[square, 'attr_str'] = attr

    return gdf1.rename(columns={'attr_str': attr_str})


@dataclass
class PixelMap:
    # initialization stuff
    voting_map: gpd.GeoDataFrame
    population_map: gpd.GeoDataFrame
    resolution: float
    # TODO: good description of what resolution means
    num_districts: int
    crs: int = 4326

    # actual contents of map
    map: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    borders: Dict[int, int] = field(default_factory=dict)

    # scoring stuff
    score: float = 0.

    surface_tension_weight: int = 3
    keep_bad_maps: float = 0.
    weights: np.ndarray = field(default_factory=lambda: np.zeros(shape=3))
    desired_results: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))

    populations: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))
    counts: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))
    centers: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))
    red_totals: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))
    blue_totals: np.ndarray = field(default_factory=lambda: np.zeros(shape=1))

    def __post_init__(self):
        # make sure both GDFs are projected into the same crs
        self.voting_map = self.voting_map.to_crs(self.crs)
        self.population_map = self.population_map.to_crs(self.crs)

        # find the area covered by at least one map
        pop_bounds = self.population_map.bounds
        pop_min = pop_bounds.min(axis=0)
        pop_max = pop_bounds.max(axis=0)
        vote_bounds = self.voting_map.bounds
        vote_min = vote_bounds.min(axis=0)
        vote_max = vote_bounds.max(axis=0)

        x_min = min(pop_min[0], vote_min[0])
        y_min = min(pop_min[1], vote_min[1])
        x_max = max(pop_max[2], vote_max[2])
        y_max = max(pop_max[3], vote_max[3])

        # determine size of squares
        x_len = x_max - x_min
        y_len = y_max - y_min
        max_len = max(x_len, y_len)
        square_len = (max_len * self.resolution) / 100

        # figure out how many squares are needed in each dimension
        num_x_squares = ceil(x_len / square_len)
        num_y_squares = ceil(y_len / square_len)

        # create the squares to cover the map
        squares = []
        for i in range(num_y_squares):
            for j in range(num_x_squares):
                x_start = j * square_len + x_min
                x_end = (j + 1) * square_len + x_min
                y_start = i * square_len + y_min
                y_end = (i + 1) * square_len + y_min

                square = Polygon([
                    (x_start, y_start),
                    (x_end, y_start),
                    (x_end, y_end),
                    (x_start, y_end)
                ])
                squares.append(square)

        # create squares on a map
        pixel_map = gpd.GeoDataFrame(geometry=squares, crs=self.crs)
        pixel_map.insert(0, 'square_num', range(len(pixel_map)))
        # calculate weighted average of square intersection
        population_map = weighted_intersection(pixel_map, self.population_map, 'population')
        red_map = weighted_intersection(pixel_map, self.voting_map, 'red_votes')
        blue_map = weighted_intersection(pixel_map, self.voting_map, 'blue_votes')
        # merge the new squares with the original GeoDataFrames
        pixel_map = population_map.merge(red_map, how='left')
        pixel_map = pixel_map.merge(blue_map, how='left')
        # clean up the resulting DataFrame
        pixel_map.set_geometry('geometry')
        pixel_map = pixel_map.dropna().reset_index()
        pixel_map.drop(columns=['index'], inplace=True)

        # add the neighbors column
        kept_squares = set(pixel_map['square_num'].unique())

        def neighbor_list(row: pd.Series) -> List[int]:
            neighbors = []
            for move in (
                    1,
                    -1,
                    num_x_squares,
                    -1 * num_x_squares
            ):
                target_idx = row.square_num + move
                if target_idx < 0 or target_idx > (num_x_squares * num_y_squares) or target_idx not in kept_squares:
                    continue
                neighbors.append(target_idx)

            return neighbors

        pixel_map['neighbors'] = pixel_map.apply(lambda row: neighbor_list(row), axis=1)
        pixel_map.set_index('square_num', inplace=True)
        self.map = pixel_map

    def initialize_districts(self) -> None:
        """This function will create the Vornoi tesselation to assign each pixel to a district.
           This function also calculates the initial border dictionary."""
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
        self.map['class'] = assignments

        # if there is at least one neighbor of a different class, mark as border in separate array
        borders = {}
        visited = [self.map.index.values[0]]
        queue = [self.map.index.values[0]]
        while queue:
            focus = queue.pop(0)
            neighbors = self.map.at[focus, 'neighbors']
            focus_class = self.map.at[focus, 'class']

            neighbor_classes = []
            for neighbor in neighbors:
                neighbor_classes.append(self.map.at[neighbor, 'class'])
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)

            different_classes = [n_class != focus_class for n_class in neighbor_classes]
            borders[focus] = sum(different_classes)

        self.borders = borders
        self.start_score()

    def start_score(self) -> None:
        """This function calculates the initial scoring variables. Necessary to build the starting scoring arrays."""
        # create variables to fill
        pops = np.zeros(self.num_districts)
        counts = np.zeros(self.num_districts)
        centers = np.zeros(shape=(self.num_districts, 2))
        red_votes = np.zeros(self.num_districts)
        blue_votes = np.zeros(self.num_districts)
        points_by_district: Dict[int, np.ndarray] = {}

        # rotate over each district
        for district in range(self.num_districts):
            # find simple sum metrics
            gdf = self.map[self.map['class'] == district]
            pop, red, blue = gdf[['population', 'red_votes', 'blue_votes']].sum()
            pops[district] = pop
            red_votes[district] = red
            blue_votes[district] = blue
            # find population center
            shapely_points = gdf['geometry'].centroid
            points = shapely_to_array(shapely_points)
            points_by_district[district] = points
            counts[district] = len(points)

            centers[district] = np.average(points, weights=gdf['population'].to_numpy(), axis=0)

        # store the pre-calculated metrics
        self.populations = pops
        self.counts = counts
        self.centers = centers
        self.red_totals = red_votes
        self.blue_totals = blue_votes

        # actually score the map
        scoring = np.zeros(3)
        # spread of population
        pops -= np.mean(pops)
        scoring[0] = np.sum(np.power(pops, 4))
        # spread of population from center
        pre_dist = [distance.cdist(points_by_district[i], centers[i].reshape((1, 2))) for i in
                    range(self.num_districts)]
        dist = [arr.flatten() for arr in pre_dist]
        scoring[1] = np.mean([np.std(arr) for arr in dist])
        # calculate deviation from election results
        current_results = red_votes / (red_votes + blue_votes)
        sorted_results = np.sort(current_results)
        # TODO: actually fit the results to some curve
        scoring[2] = np.std(sorted_results)

        score = self.weights @ scoring.T
        self.score = score

    def evaluate(self, pop, center, red, blue) -> float:
        """This function scores a map that doesn't actually exist. The scoring arrays are passed in and the evaluation
           is returned

           :param pop: The population scoring array containing populations of each district
           :param center: The center scoring array containing centers of each district
           :param red: The red votes scoring array containing the total number of red votes in each district
           :param blue: The blue votes scoring array containing the total number of blue votes in each district

           :return the score represented by a float
        """
        # actually score the map
        scoring = np.zeros(3)
        # spread of population
        pop -= np.mean(pop)
        scoring[0] = np.sum(np.power(pop, 4))
        # spread of population from center
        # TODO: actually implement the squareness metric
        '''
        dist = [0 for _ in range(self.num_districts)]
        for idx, row in self.map.iterrows():

        pre_dist = [distance.cdist(points_by_district[i], centers[i].reshape((1, 2))) for i in
                    range(self.num_districts)]
        dist = [arr.flatten() for arr in pre_dist]
        scoring[1] = np.mean([np.std(arr) for arr in dist])
        '''
        scoring[1] = 1
        # calculate deviation from election results
        current_results = red / (red + blue)
        sorted_results = np.sort(current_results)
        # TODO: actually fit the results to some curve
        scoring[2] = np.std(sorted_results)

        return self.weights @ scoring.T

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
            neighbor_class = self.map.at[neighbor, 'class']
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
                            self.map.at[focus_neighbor, 'class'] == class_num and
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
        remove_class = self.map.at[remove_pixel, 'class']

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
            first_class = self.map.at[first_pixel, 'class']
            neighbors = self.map.at[first_pixel, 'neighbors']
            random.shuffle(neighbors)

            for neighbor in neighbors:
                second_class = self.map.at[neighbor, 'class']
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
        # swap classes
        original_class = self.map.at[first, 'class']
        destination_class = self.map.at[second, 'class']
        self.map.at[first, 'class'] = destination_class

        # update metrics
        pop = self.map.at[first, 'population']
        red = self.map.at[first, 'red_votes']
        blue = self.map.at[first, 'blue_votes']
        # population
        origin_new_pop = self.populations[original_class] - pop
        destination_new_pop = self.populations[destination_class] + pop
        # centers
        # this is fairly easy to follow if you remove an arbitrary item from an average
        # back of the napkin math shows the formula
        # the relative unreadability is just because of how funky it is to type out math
        c = self.populations[original_class]
        n = self.counts[original_class]
        pixel_point = self.map.at[first, 'geometry'].centroid
        pixel_arr = np.array([pixel_point.x, pixel_point.y])

        origin_new_center = ((c*n) - pixel_arr) / (n - 1)

        d = self.populations[destination_class]
        n = self.counts[destination_class]

        destination_new_center = ((d*n) + pixel_arr) / (n + 1)
        # results
        origin_new_red = self.red_totals[original_class] - red
        origin_new_blue = self.blue_totals[original_class] - blue
        destination_new_red = self.red_totals[destination_class] + red
        destination_new_blue = self.red_totals[destination_class] + blue

        # score itself
        # duplicate the arrays
        n_populations = self.populations.copy()
        n_populations[original_class] = origin_new_pop
        n_populations[destination_class] = destination_new_pop

        n_centers = self.centers.copy()
        n_centers[original_class] = origin_new_center
        n_centers[destination_class] = destination_new_center

        n_red = self.red_totals.copy()
        n_red[original_class] = origin_new_red
        n_red[destination_class] = destination_new_red

        n_blue = self.blue_totals.copy()
        n_blue[original_class] = origin_new_blue
        n_blue[destination_class] = destination_new_blue

        new_score = self.evaluate(n_populations, n_centers, n_red, n_blue)

        if new_score > self.score and random.random() < self.keep_bad_maps:
            self.map.at[first, 'class'] = original_class
            return False

        # update the internal fields
        self.counts[original_class] -= 1
        self.counts[destination_class] += 1

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
            f_neighbor_class = [self.map.at[f_neighbor, 'class'] for f_neighbor in focus_neighbors]
            pixel_class = self.map.at[pixel, 'class']
            different_classes = [n_class != pixel_class for n_class in f_neighbor_class]
            self.borders[pixel] = sum(different_classes)

        return True

    def show_districts(self):
        """This function plots a choropleth of the internal GeoDataFrame with the district as the color var."""
        self.map.plot(column='class')
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
    pixel_map.map.plot(column='class', ax=ax)
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
