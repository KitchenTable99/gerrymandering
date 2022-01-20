# this file will merge ACS and TIGER data to create the location-based population file
import argparse
import glob
import os
from zipfile import ZipFile
from tqdm import tqdm as progress

import geopandas as gpd
import pandas as pd
import requests

with open('census_api_key_secret.txt', 'r') as fp:
    API_KEY = fp.read()[:-1]

CENSUS_COLUMN = 'P1_001N'

STATE_ABBREV = {'oh', 'ms', 'ny', 'ky', 'or', 'nv', 'wi', 'md', 'in', 'ct', 'ks', 'nd', 'sc', 'tn', 'ca', 'va', 'me',
                'sd', 'nm', 'la', 'dc', 'ok', 'mi', 'ri', 'ga', 'mn', 'ne', 'al', 'nh', 'mt', 'wv', 'fl', 'hi', 'ia',
                'pa', 'ar', 'nj', 'az', 'ma', 'il', 'nc', 'mo', 'ut', 'wa', 'ak', 'de', 'id', 'tx', 'co', 'vt', 'wy',
                'all'}

STATE_DICT = {'al': 1, 'ak': 2, 'az': 4, 'ar': 5, 'ca': 6, 'co': 8, 'ct': 9, 'de': 10, 'id': 16, 'fl': 12, 'ga': 13,
              'hi': 15, 'il': 17, 'in': 18, 'ia': 19, 'ks': 20, 'ky': 21, 'la': 22, 'me': 23, 'md': 24, 'ma': 25,
              'mi': 26, 'mn': 27, 'ms': 28, 'mo': 29, 'mt': 30, 'ne': 31, 'nv': 32, 'nh': 33, 'nj': 34, 'nm': 35,
              'ny': 36, 'nc': 37, 'nd': 38, 'oh': 39, 'ok': 40, 'or': 41, 'pa': 42, 'ri': 44, 'sc': 45, 'sd': 46,
              'tn': 47, 'tx': 48, 'ut': 49, 'vt': 50, 'va': 51, 'wa': 53, 'wv': 54, 'wi': 55, 'wy': 56}


def get_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Retrieve population data from the Census Bureau at the block level '
                                                 'and joins it with shapefile data contained within the data '
                                                 'directory.')
    parser.add_argument('state', type=str, choices=STATE_ABBREV, help='The state for which to get data. If all is '
                                                                      'passed, all 50 states will be queried')

    return parser.parse_args()


def get_census(state_id: int) -> pd.DataFrame:
    """
    :param state_id: the number of the state according to the Census Bureau classification.
    :return: a DataFrame containing the population data and the GeoID of the passed state
    """
    url = f'https://api.census.gov/data/2020/dec/pl?' \
          f'get={CENSUS_COLUMN},GEO_ID&for=block:*&in=state:{state_id:02}&in=county:*&in=tract:*&key={API_KEY}'

    response = requests.get(url)
    response_json = response.json()

    return pd.DataFrame(response_json[1:], columns=response_json[0])


def main():
    cmd_args = get_cmd_args()

    if cmd_args.state == 'all':
        get_all_pop()
    else:
        state_num = STATE_DICT.get(cmd_args.state)
        driver(cmd_args.state, state_num)


def get_all_pop():
    for state, state_num in progress(STATE_DICT.items()):
        driver(state, state_num, verbose=False)


def driver(state: str, state_num: int, verbose: bool = True):
    # population
    if verbose:
        print(f'Getting {state.upper()} population data...')
    population_df = get_census(state_num)
    # clean up population data
    population_df['geo_id'] = population_df.apply(lambda row: row.GEO_ID.split('US')[-1], axis=1)
    population_df = population_df[['geo_id', CENSUS_COLUMN]]

    # TIGER
    if verbose:
        print('Unzipping shapefile data...')
    tiger_template = f'tl_2021_{state_num:02}_tabblock20'
    tiger_zip = f'./data/tiger_shapefiles/{tiger_template}.zip'
    with ZipFile(tiger_zip, 'r') as zip_:
        zip_.extractall()
    if verbose:
        print('Reading shapefile data...')
    tiger_df = gpd.read_file(tiger_template + '.shp')
    # clean up TIGER data
    tiger_df = tiger_df[['GEOID20', 'geometry']]
    tiger_df.rename(columns={'GEOID20': 'geo_id'}, inplace=True)

    # merge and clean
    if verbose:
        print('Merging and saving data...')
    merged = tiger_df.merge(population_df, on='geo_id')
    merged.rename(columns={CENSUS_COLUMN: 'population'}, inplace=True)
    merged = merged.astype({'population': int})

    # save file
    save_path = f'./data/population/{state}.geojson'
    merged.to_file(save_path, driver='GeoJSON')

    # delete all the tiger stuff
    tiger_glob = glob.glob(f'{tiger_template}.*')
    for file in tiger_glob:
        os.remove(file)


if __name__ == '__main__':
    main()
