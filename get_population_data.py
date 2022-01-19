# this file will merge ACS and TIGER data to create the location-based population file
import glob
import os
import sys
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests

with open('census_api_key_secret.txt', 'r') as fp:
    API_KEY = fp.read()[:-1]

CENSUS_COLUMN = 'P1_001N'

STATE_DICT = {'Alabama': 1, 'Alaska': 2, 'Arizona': 4, 'Arkansas': 5, 'California': 6, 'Colorado': 8, 'Connecticut': 9,
              'Delaware': 10, 'District of Columbia': 11, 'Idaho': 16, 'Florida': 12, 'Georgia': 13, 'Hawaii': 15,
              'Illinois': 17, 'Indiana': 18, 'Iowa': 19, 'Kansas': 20, 'Kentucky': 21, 'Louisiana': 22, 'Maine': 23,
              'Maryland': 24, 'Massachusetts': 25, 'Michigan': 26, 'Minnesota': 27, 'Mississippi': 28, 'Missouri': 29,
              'Montana': 30, 'Nebraska': 31, 'Nevada': 32, 'New Hampshire': 33, 'New Jersey': 34, 'New Mexico': 35,
              'New York': 36, 'North Carolina': 37, 'North Dakota': 38, 'Ohio': 39, 'Oklahoma': 40, 'Oregon': 41,
              'Pennsylvania': 42, 'Rhode Island': 44, 'South Carolina': 45, 'South Dakota': 46, 'Tennessee': 47,
              'Texas': 48, 'Utah': 49, 'Vermont': 50, 'Virginia': 51, 'Washington': 53, 'West Virginia': 54,
              'Wisconsin': 55, 'Wyoming': 56, 'Puerto Rico': 72}

STATE_ABBREV = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
                'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
                'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
                'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
                'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
                'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
                'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
                'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}


def usage_statement() -> None:
    """This just prints the usage statement and exits."""
    print('\nUSAGE STATEMENT:')
    print(f'python3 {sys.argv[0]} [STATE NAME]')
    print('\n\nThis file takes in the name of a United State and processes it to get its population data at the census'
          'block level. The name of the state must be title cased (e.g. North Carolina, California). '
          'It writes this data to a geojson file to the current working directory.')
    sys.exit()


def get_census(state_id: int) -> pd.DataFrame:
    """
    :param state_id: the number of the state according to the Census Bureau classification.
    :return: a DataFrame containing the population data and the GeoID of the passed state
    """
    url = f'https://api.census.gov/data/2020/dec/pl?' \
          f'get={CENSUS_COLUMN},GEO_ID&for=block:*&in=state:{state_id}&in=county:*&in=tract:*&key={API_KEY}'

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
    population_df = population_df[['geo_id', CENSUS_COLUMN]]

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
    merged.rename(columns={CENSUS_COLUMN: 'population'}, inplace=True)
    merged = merged.astype({'population': int})

    # save file
    save_path = f'./data/population/{STATE_ABBREV.get(state).lower()}.geojson'
    merged.to_file(save_path, driver='GeoJSON')

    # delete all the tiger stuff
    tiger_glob = glob.glob(f'{tiger_template}.*')
    for file in tiger_glob:
        os.remove(file)


if __name__ == '__main__':
    main()
