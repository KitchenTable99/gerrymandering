# this file contains code needed to create a pixel map GeoDataFrame
from math import ceil
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


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


def make_pixel_map(voting_map: gpd.GeoDataFrame, population_map: gpd.GeoDataFrame, resolution: float, verbose: bool = False) -> gpd.GeoDataFrame:
    # make sure both GDFs are projected into the same crs
    if verbose:
        print('Projecting maps...')
    if population_map.crs != 4269:
        population_map = population_map.to_crs(4269)
    if voting_map.crs != 4269:
        voting_map = voting_map.to_crs(4269)

    # find the area covered by at least one map
    pop_bounds = population_map.bounds
    pop_min = pop_bounds.min(axis=0)
    pop_max = pop_bounds.max(axis=0)
    vote_bounds = voting_map.bounds
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
    square_len = (max_len * resolution) / 100

    # figure out how many squares are needed in each dimension
    num_x_squares = ceil(x_len / square_len)
    num_y_squares = ceil(y_len / square_len)

    if verbose:
        print('Creating squares...')
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
    pixel_map = gpd.GeoDataFrame(geometry=squares, crs=4269)
    pixel_map.insert(0, 'square_num', range(len(pixel_map)))
    # calculate weighted average of square intersection
    if verbose:
        print('Overlaying population map...')
    population_map = weighted_intersection(pixel_map, population_map, 'population')
    if verbose:
        print('Overlaying voting map (red_votes)...')
    red_map = weighted_intersection(pixel_map, voting_map, 'red_votes')
    if verbose:
        print('Overlaying voting map (blue_votes)...')
    blue_map = weighted_intersection(pixel_map, voting_map, 'blue_votes')
    # merge the new squares with the original GeoDataFrames
    if verbose:
        print('Merging maps...')
    pixel_map = population_map.merge(red_map, how='left')
    pixel_map = pixel_map.merge(blue_map, how='left')
    # clean up the resulting DataFrame
    pixel_map.set_geometry('geometry')
    pixel_map = pixel_map.dropna().reset_index()
    pixel_map.drop(columns=['index'], inplace=True)

    # add the neighbors column
    kept_squares = set(pixel_map['square_num'].unique())
    if verbose:
        print('Finding neighbors...')

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

    return pixel_map


def add_numpy_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf['np_geometry'] = gdf.apply(lambda row: np.array((row['geometry'].centroid.x, row['geometry'].centroid.y)),
                                   axis=1)

    return gdf


def main():
    nc_votes = gpd.read_file('./nc_data/voters_shapefile/NC_G18.shp')
    nc_votes = nc_votes[['G18DStSEN', 'G18RStSEN', 'geometry']].rename(
        columns={'G18DStSEN': 'blue_votes', 'G18RStSEN': 'red_votes'})

    nc_pop = gpd.read_file('./nc_data/population.geojson')

    nc_map = make_pixel_map(nc_votes, nc_pop, .1, verbose=True)
    nc_map = add_numpy_geometry(nc_map)
    nc_map.to_pickle('full_map.pickle')


if __name__ == '__main__':
    main()