from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd


def get_data():
    input_path = '../datasets/flight_fare.csv'
    df = pd.read_csv(input_path, sep=';', decimal=',')
    return df


def get_range(value, range_size, min_val, max_val):
    if range_size == 1:
        return f'[{value}-{value}]'

    range_num = (value - 1) // range_size
    lower = max(min_val, (range_num * range_size) + 1)
    upper = min(max_val, ((range_num + 1) * range_size))

    return f'[{lower}-{upper}]'


def get_date_num_hierarchy(df, col_name):
    range_sizes = [3, 6, 12, 36, 72, 144]
    hierarchy = pd.DataFrame(dtype='<U32')

    hierarchy[0] = df[f'{col_name} num']

    date_min = df[f'{col_name} num'].min()
    date_max = df[f'{col_name} num'].max()
    date_values = np.arange(date_min, date_max + 1)
    get_range_vect = np.vectorize(get_range, otypes='O')
    for index, range_size in enumerate(range_sizes):
        ranges_str = get_range_vect(date_values, range_size, date_min, date_max)
        output = ranges_str[df[f'{col_name} num'].values - date_min]
        hierarchy[index + 1] = output
    hierarchy[index + 2] = f'[{date_min}-{date_max}]'
    return hierarchy


def get_time_hierarchy(df):
    range_sizes = [3, 6, 12, 36, 72, 144]
    hierarchy = pd.DataFrame(dtype='<U32')

    hierarchy[0] = df['Duration']
    time_min = df['Duration'].min()
    time_max = df['Duration'].max()
    time_values = np.arange(time_min, time_max + 1)
    get_range_vect = np.vectorize(get_range, otypes='O')
    for index, range_size in enumerate(range_sizes):
        ranges_str = get_range_vect(time_values, range_size, time_min, time_max)
        output = ranges_str[df['Duration'].values - time_min]
        hierarchy[index + 1] = output
    hierarchy[index + 2] = f'[{time_min}-{time_max}]'
    return hierarchy


if __name__ == '__main__':
    out_path = '../traditional_hierarchies/'
    print('reading data')
    df = get_data()

    print('creating duration hierarchy')
    hierarchy = get_time_hierarchy(df)
    hierarchy.to_csv(f'{out_path}Duration.csv', sep=';', decimal=',', index=False, header=False)

    print('creating departure time hierarchy')
    hierarchy = get_date_num_hierarchy(df, 'Departure Time')
    hierarchy.to_csv(f'{out_path}Departure Time.csv', sep=';', decimal=',', index=False, header=False)

    print('creating arrival time hierarchy')
    hierarchy = get_date_num_hierarchy(df, 'Arrival Time')
    hierarchy.to_csv(f'{out_path}Arrival Time.csv', sep=';', decimal=',', index=False, header=False)

