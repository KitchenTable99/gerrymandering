# the code to visualize map change

import geopandas as gpd
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation import *


@dataclass
class PixelMapAnimator:
    map: PixelMap
    fig: plt.Figure
    ax: plt.Axes
    frame_gap: int

    def __post_init__(self):
        self.ax.axis('off')

    def __call__(self, i: int) -> None:
        if i == 0:
            self.map.map.plot(ax=self.ax, column='class')
            return self.ax.get_children()

        for _ in range(self.frame_gap):
            first, second = self.map.pick_swap_pair()
            self.map.swap_pixels(first, second)
        self.map.map.plot(ax=self.ax, column='class')
        return self.ax.get_children()


def time():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)
    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(pixel_map, fig, ax, 1)
    import timeit

    # print(timeit.timeit(lambda: animator(1), number=10))
    # sys.exit()
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        animator(1)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profile.prof')


def save():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    fig, ax = plt.subplots(1)
    pixel_map.weights = np.array([10, 20, 0])
    animator = PixelMapAnimator(pixel_map, fig, ax, 1000)
    anim = FuncAnimation(fig, animator, frames=50, interval=500)
    anim.save('testing.gif', writer='imagemagick', fps=10)


def main():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    fig, ax = plt.subplots(1)
    pixel_map.weights = np.array([10, 20, 0])
    animator = PixelMapAnimator(pixel_map, fig, ax, 1000)
    anim = FuncAnimation(fig, animator, frames=100, interval=1000)
    plt.show()


if __name__ == '__main__':
    save()
    # main()
    # time()
