from shapely.geometry import Point
import pandas as pd
from typing import List, Generator, Dict
import numpy as np

# TODO: this doesn't live here
def pixel_to_center(pixel_centroid: Point, district: 'District') -> float:
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


def shapely_to_array(points: List[Point]) -> np.ndarray:
    """This function converts a list of shapely points into an array of points to be used in cdist.

        :param points. A list of points from the shapely library

        :returns an array of points
    """
    point_tuples = [(point.x, point.y) for point in points]

    return np.array([*point_tuples])
