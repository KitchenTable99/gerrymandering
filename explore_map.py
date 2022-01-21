import geopandas as gpd
import pickle
import matplotlib.pyplot as plt

with open('simulation_out.pickle', 'rb') as fp:
	df = pickle.load(fp)

df.plot(column='district', cmap='tab20')
plt.show()
