# this file contains the classes relevant to scoring
from typing import NamedTuple
import numpy as np


class WeightValues(NamedTuple):
    statistical_surface_tension: int
    scoring_weights: np.ndarray
    keep_bad_maps: float


class WeightDict(NamedTuple):
    centering: WeightValues
    exploring: WeightValues
    electioneering: WeightValues


def main():
    wd = WeightDict(WeightValues(3, np.diag((10, 20, 3)), .5), WeightValues(1, np.diag((1, 1, 1)), .75),
                    WeightValues(3, np.diag((1, 2, 30)), .5))

    print(wd.centering)


if __name__ == '__main__':
    main()
