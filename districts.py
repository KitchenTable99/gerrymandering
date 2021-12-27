# this holds the code related to district objects
from dataclasses import dataclass
from math import dist
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

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

    def add_deviation(self, geom: Point) -> None:
        point_tuple = (geom.x, geom.y)
        self.deviation += dist(point_tuple, self.population_center)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> 'District':
        # find simple sum metrics
        pop, red, blue = df[['population', 'red_votes', 'blue_votes']].sum()

        # find population center
        shapely_points = df['geometry'].centroid
        points = shapely_to_array(shapely_points)
        center = np.average(points, weights=df['population'].to_numpy(), axis=0)

        return District(pop, red, blue, center)
