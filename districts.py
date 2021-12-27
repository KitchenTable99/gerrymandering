# this holds the code related to district objects
from dataclasses import dataclass
from math import dist
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from utilities import shapely_to_array


@dataclass
class District:

    population: float
    red_votes: float
    blue_votes: float
    population_center: np.ndarray
    deviation: float = 0.

    @property
    def election_result(self) -> float:
        return self.red_votes / self.blue_votes

    def add_deviation(self, geom: Polygon) -> None:
        centroid = geom.centroid
        point_tuple = (centroid.x, centroid.y)
        self.deviation += dist(point_tuple, self.population_center)

    def reset_deviation(self) -> None:
        self.deviation = 0

    def add_pixel(self, pixel_data: pd.Series) -> None:
        # simple sums
        new_population = self.population + pixel_data['population']
        self.red_votes += pixel_data['red_votes']
        self.blue_votes += pixel_data['blue_votes']

        # population center
        # this seems like crazy math, but back-of-the-napkin calculations will show how to add an item
        # from a weighted average
        centroid = pixel_data['geometry'].centroid
        pixel_center = np.array((centroid.x, centroid.y))
        self.population_center *= self.population
        self.population_center += pixel_center * pixel_data['population']
        self.population_center /= new_population

        self.population = new_population

    def remove_pixel(self, pixel_data: pd.Series) -> None:
        # simple sums
        new_population = self.population - pixel_data['population']
        self.red_votes -= pixel_data['red_votes']
        self.blue_votes -= pixel_data['blue_votes']

        # population center
        # this seems like crazy math, but back-of-the-napkin calculations will show how to remove an item
        # from a weighted average
        centroid = pixel_data['geometry'].centroid
        pixel_center = np.array((centroid.x, centroid.y))
        self.population_center *= self.population
        self.population_center -= pixel_center * pixel_data['population']
        self.population_center /= new_population

        self.population = new_population

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> 'District':
        # find simple sum metrics
        pop, red, blue = df[['population', 'red_votes', 'blue_votes']].sum()

        # find population center
        shapely_points = df['geometry'].centroid
        points = shapely_to_array(shapely_points)
        center = np.average(points, weights=df['population'].to_numpy(), axis=0)

        return District(pop, red, blue, center)
