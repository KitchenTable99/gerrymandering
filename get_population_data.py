# this file will merge ACS and TIGER data to create the location-based population file
import argparse
import glob
import os
from zipfile import ZipFile
from tqdm import tqdm as progress

import geopandas as gpd
import pandas as pd
import requests



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
    parser.add_argument('year', type=int, choices={2010, 2020}, help='Which census year do you want data from?')

    return parser.parse_args()


def get_census(year: int, column: str, state_id: int, api_key: str) -> pd.DataFrame:
    """
    :param state_id: the number of the state according to the Census Bureau classification.
    :return: a DataFrame containing the population data and the GeoID of the passed state
    """
    url = f'https://api.census.gov/data/{year}/dec/pl?' \
          f'get={column},GEO_ID&for=block:*&in=state:{state_id:02}&in=county:*&in=tract:*&key={api_key}'

    response = requests.get(url)
    response_json = response.json()

    return pd.DataFrame(response_json[1:], columns=response_json[0])


def main():
    cmd_args = get_cmd_args()

    if cmd_args.state == 'all':
        get_all_pop(cmd_args.year)
    else:
        state_num = STATE_DICT.get(cmd_args.state)
        driver(cmd_args.state, state_num, cmd_args.year)


def get_all_pop(year: int):
    for state, state_num in progress(STATE_DICT.items()):
        if os.path.exists(f'./data/population/{year}/{state}.geojson'):
            continue
        else:
            driver(state, state_num, year, verbose=False)


def driver(state: str, state_num: int, year: int, verbose: bool = True):
    # population
    if verbose:
        print(f'Getting {state.upper()} population data...')
    # set up parameters
    with open('census_api_key_secret.txt', 'r') as fp:
        api_key = fp.read()[:-1]

    column = 'P001001' if year == 2010 else 'P1_001N'
    population_df = get_census(year, column, state_num, api_key)
    # clean up population data
    population_df['geo_id'] = population_df.apply(lambda row: row.GEO_ID.split('US')[-1], axis=1)
    population_df = population_df[['geo_id', column]]

    # TIGER
    if verbose:
        print('Unzipping shapefile data...')
    tiger_template = f'tl_{year}_{state_num:02}_tabblock{year % 100}'  # nb: year % 100 returns the last two digits
    tiger_zip = f'./data/tiger_shapefiles/{year}/{tiger_template}.zip'
    with ZipFile(tiger_zip, 'r') as zip_:
        zip_.extractall()
    if verbose:
        print('Reading shapefile data...')
    tiger_df = gpd.read_file(tiger_template + '.shp')
    # clean up TIGER data
    returned_geoid = 'GEOID10' if year == 2010 else 'GEOID20'
    tiger_df = tiger_df[[f'{returned_geoid}', 'geometry']]
    tiger_df.rename(columns={f'{returned_geoid}': 'geo_id'}, inplace=True)

    # merge and clean
    if verbose:
        print('Merging and saving data...')
    merged = tiger_df.merge(population_df, on='geo_id')
    merged.rename(columns={column: 'population'}, inplace=True)
    merged = merged.astype({'population': int})

    # save file
    save_path = f'./data/population/{year}/{state}.geojson'
    merged.to_file(save_path, driver='GeoJSON')

    # delete all the tiger stuff
    tiger_glob = glob.glob(f'{tiger_template}.*')
    for file in tiger_glob:
        os.remove(file)


if __name__ == '__main__':
    main()
