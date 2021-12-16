# the code to visualize map change

import geopandas as gpd
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pixel_classes import *


@dataclass
class PixelMapAnimator:
    map: PixelMap
    fig: plt.Figure
    ax: plt.Axes

    def __call__(self, i: int) -> None:
        if i == 0:
            self.map.map.plot(ax=self.ax, column='class')
            return

        first, second = self.map.pick_swap_pair()
        self.map.swap_pixels(first, second)
        self.map.map.plot(ax=self.ax, column='class')
        return


def time():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)
    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(pixel_map, fig, ax)
    import timeit

    print(timeit.timeit(lambda: animator(1), number=100))


def save():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(pixel_map, fig, ax)
    anim = FuncAnimation(fig, animator, frames=180, interval=500)
    anim.save('testing.gif', writer='imagemagick', fps=60)


def main():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(pixel_map, fig, ax)
    anim = FuncAnimation(fig, animator, frames=180, interval=500)
    plt.show()


if __name__ == '__main__':
    # save()
    main()
    # time()
