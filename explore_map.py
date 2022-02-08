from typing import Optional

import geopandas as gpd
import pickle
import matplotlib.pyplot as plt

with open('simulation_out.pickle', 'rb') as fp:
	df = pickle.load(fp)


def pop_bar(ax: Optional[plt.Axes] = None) -> None:
	pop = df.groupby('district').sum().population
	print(f'Biggest difference = {pop.max() - pop.min()}\nSpread = {pop.std()}')
	if ax:
		ax.bar(range(len(pop)), pop)
	else:
		plt.bar(range(len(pop)), pop)
		plt.show()


fig, ax = plt.subplots(2)
df.plot(column='district', ax=ax[0], cmap='tab20')
pop_bar(ax[1])
plt.show()


def plot_district(to_plot: int) -> None:
	map_plot = df[df['district'] == to_plot]
	not_map_plot = df[df['district'] != to_plot]
	fig_, ax_ = plt.subplots()
	not_map_plot.plot(color='black', ax=ax_)
	map_plot.plot(color='red', ax=ax_)
	plt.show()
