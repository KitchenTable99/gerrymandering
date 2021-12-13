# this file will merge ACS and TIGER data to create the location-based population file

import sys
import pandas as pd
import geopandas as gpd


def usage_statement() -> None:
    """This just prints the usage statement and exits."""
    print('\nUSAGE STATEMENT:')
    print(f'python3 {sys.argv[0]} [SHAPEFILE] [ACS FILE]')
    print('\n\nThis file takes in a TIGER shapefile as well as a CSV file downloaded from socialexplorer.com and writes'
          ' a geojson file to the current working directory.')
    sys.exit()


def main():
    # get the input files
    # must be according to usage statement
    try:
        argv = sys.argv
        _, tiger, acs = argv
        if not tiger.endswith('.shp') or not acs.endswith('.csv'):
            usage_statement()

        tiger_df = gpd.read_file(tiger)
        acs_df = pd.read_csv(acs)
    except ValueError:
        usage_statement()

    # prep merge
    tiger_df.rename(columns={'GEOID20': 'Geo_FIPS'}, inplace=True)
    tiger_df = tiger_df.astype({'Geo_FIPS': 'int32'})

    # merge
    merged = tiger_df.merge(acs_df, on='Geo_FIPS')
    merged.rename(columns={'SE_A00001_001': 'population'}, inplace=True)

    # keep needed columns
    population_df = merged[['geometry', 'population']]

    # save file
    population_df.to_file('population.geojson', driver='GeoJSON')


if __name__ == '__main__':
    main()