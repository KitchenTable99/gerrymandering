# this file will merge ACS and TIGER data to create the location-based population file
import os
import sys
import requests
import glob
import pandas as pd
from zipfile import ZipFile
import geopandas as gpd

API_KEY = 'ce766d0c0c87a250b8ea638692c417d1b7094fc3'
STATE_DICT = {'Alabama': 1, 'Alaska': 2, 'Arizona': 4, 'Arkansas': 5, 'California': 6, 'Colorado': 8, 'Connecticut': 9,
              'Delaware': 10, 'District of Columbia': 11, 'Idaho': 16, 'Florida': 12, 'Georgia': 13, 'Hawaii': 15,
              'Illinois': 17, 'Indiana': 18, 'Iowa': 19, 'Kansas': 20, 'Kentucky': 21, 'Louisiana': 22, 'Maine': 23,
              'Maryland': 24, 'Massachusetts': 25, 'Michigan': 26, 'Minnesota': 27, 'Mississippi': 28, 'Missouri': 29,
              'Montana': 30, 'Nebraska': 31, 'Nevada': 32, 'New Hampshire': 33, 'New Jersey': 34, 'New Mexico': 35,
              'New York': 36, 'North Carolina': 37, 'North Dakota': 38, 'Ohio': 39, 'Oklahoma': 40, 'Oregon': 41,
              'Pennsylvania': 42, 'Rhode Island': 44, 'South Carolina': 45, 'South Dakota': 46, 'Tennessee': 47,
              'Texas': 48, 'Utah': 49, 'Vermont': 50, 'Virginia': 51, 'Washington': 53, 'West Virginia': 54,
              'Wisconsin': 55, 'Wyoming': 56, 'Puerto Rico': 72}


def usage_statement() -> None:
    """This just prints the usage statement and exits."""
    print('\nUSAGE STATEMENT:')
    print(f'python3 {sys.argv[0]} [SHAPEFILE] [ACS FILE]')
    print('\n\nThis file takes in a TIGER shapefile as well as a CSV file downloaded from socialexplorer.com and writes'
          ' a geojson file to the current working directory.')
    sys.exit()


def get_census(state_id: int) -> pd.DataFrame:
    url = f'https://api.census.gov/data/2020/dec/pl?' \
          f'get=P3_001N,GEO_ID&for=block:*&in=state:{state_id}&in=county:*&in=tract:*&key={API_KEY}'

    response = requests.get(url)
    response_json = response.json()

    return pd.DataFrame(response_json[1:], columns=response_json[0])


def main():
    # get the input files
    # must be according to usage statement
    state_num = None
    state = None
    try:
        state = ' '.join(sys.argv[1:])
        state_num = STATE_DICT.get(state, None)
    except IndexError:
        usage_statement()

    if not state_num:
        usage_statement()

    # population
    print(f'Getting {state} population data...')
    population_df = get_census(state_num)
    # clean up population data
    population_df['geo_id'] = population_df.apply(lambda row: row.GEO_ID.split('US')[-1], axis=1)
    population_df = population_df[['geo_id', 'P3_001N']]

    # TIGER
    print('Unzipping shapefile data...')
    tiger_template = f'tl_2021_{state_num:02}_tabblock20'
    tiger_zip = f'./data/tiger_shapefiles/{tiger_template}.zip'
    with ZipFile(tiger_zip, 'r') as zip_:
        zip_.extractall()
    print('Reading shapefile data...')
    tiger_df = gpd.read_file(tiger_template + '.shp')
    # clean up TIGER data
    tiger_df = tiger_df[['GEOID20', 'geometry']]
    tiger_df.rename(columns={'GEOID20': 'geo_id'}, inplace=True)

    # merge and clean
    print('Merging and saving data...')
    merged = tiger_df.merge(population_df, on='geo_id')
    merged.rename(columns={'P3_001N': 'population'}, inplace=True)

    # save file
    merged.to_file(f'{state.lower().replace(" ", "_")}_population.geojson', driver='GeoJSON')

    # delete all the tiger stuff
    tiger_glob = glob.glob(f'{tiger_template}.*')
    for file in tiger_glob:
        os.remove(file)


if __name__ == '__main__':
    main()
