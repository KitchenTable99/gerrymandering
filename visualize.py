# the code to visualize map change
from typing import List, Tuple

import geopandas as gpd
import sys
from dataclasses import dataclass, field
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
with open('log.pickle', 'rb') as fp:
    MAP = pickle.load(fp)
LOGS = pd.read_csv('log_changes.csv')
from districts import District
from simulation import GerrymanderingSimulation
Switch = Tuple[int, int]  # the first int is the square_num, the second is the class it switched from


@dataclass(unsafe_hash=True)
class LoggingMapAnimator:

    switches: List[Switch] = field(default_factory=list)
    switches_done: int = 0

    def move_forward(self, map_: gpd.GeoDataFrame, logs: pd.DataFrame, switch_num: int) -> None:
        to_switch = logs.iloc[switch_num]
        current_district = map_.at[to_switch.square_num, 'district']
        map_.at[to_switch.square_num, 'district'] = to_switch.switch_district
        switch = (to_switch.square_num, current_district)
        self.switches.append(switch)

    def move_backward(self, map_: gpd.GeoDataFrame) -> None:
        if self.switches:
            to_switch = self.switches.pop()
            map_.at[to_switch[0], 'district'] = to_switch[1]

    def on_press(self, event) -> None:
        if event.key == 'j':
            self.move_forward(MAP, LOGS, self.switches_done)
            self.switches_done += 1
            self.fig.canvas.draw()
        if event.key == 'k':
            self.move_backward(MAP)
            self.fig.canvas.draw()

    def plot(self, map_) -> None:
        fig, ax = plt.subplots()
        self.fig = fig
        map_.plot(column='district', ax=ax)
        fig.canvas.mpl_connect('key_press_event', self.on_press)
        plt.show()


@dataclass
class PixelMapAnimator:
    map: GerrymanderingSimulation
    fig: plt.Figure
    ax: plt.Axes
    frame_gap: int

    def __post_init__(self):
        self.ax.axis('off')

    def __call__(self, i: int) -> None:
        if i == 0:
            self.map.map.plot(ax=self.ax, column='district')
            return self.ax.get_children()
        elif i <= 50:
            self.map.set_centering_weights()
            for _ in range(self.frame_gap):
                first, second = self.map.pick_swap_pair()
                self.map.swap_pixels(first, second)
            self.map.map.plot(ax=self.ax, column='district')
        elif i <= 80:
            self.map.set_exploring_weights()
            for _ in range(self.frame_gap):
                first, second = self.map.pick_swap_pair()
                self.map.swap_pixels(first, second)
            self.map.map.plot(ax=self.ax, column='district')
        else:
            self.map.set_electioneering_weights()
            for _ in range(self.frame_gap):
                first, second = self.map.pick_swap_pair()
                self.map.swap_pixels(first, second)
            self.map.map.plot(ax=self.ax, column='district')

        return self.ax.get_children()


def time():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)
    sim = GerrymanderingSimulation(pixel_map, 13)
    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(sim, fig, ax, 1)
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
    with open('test_map.pickle', 'rb') as fp:
        pixel_map: gpd.GeoDataFrame = pickle.load(fp)

    sim = GerrymanderingSimulation(pixel_map, 13)

    sim.set_desired_results(np.repeat(.6, 13))
    sim.initialize_districts()

    fig, ax = plt.subplots(1)
    animator = PixelMapAnimator(sim, fig, ax, 100)
    anim = FuncAnimation(fig, animator, frames=100, interval=450)
    anim.save('testing.gif', writer='imagemagick', fps=10)


def main():
    with open('pix.pickle', 'rb') as fp:
        pixel_map: PixelMap = pickle.load(fp)

    fig, ax = plt.subplots(1)
    pixel_map.weights = np.array([10, 20, 0])
    animator = PixelMapAnimator(pixel_map, fig, ax, 100)
    anim = FuncAnimation(fig, animator, frames=100, interval=1000)
    plt.show()

def logging():
    log_animator = LoggingMapAnimator()
    log_animator.plot(MAP)


if __name__ == '__main__':
    # save()
    # main()
    # time()
    logging()
