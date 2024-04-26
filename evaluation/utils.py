import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from flight_fare.analysis.queries import hull_generator as hg


def parse_original_data(row):
    entry = row.split('::')
    values = [int(data) for data in entry]
    return tuple(values)


def parse_column(row):
    ranges_str = row.split('::')
    output = []
    for r_str in ranges_str:
        try:
            val = int(r_str)
            output.append([val, val])
        except ValueError:
            if '[' in r_str and ']' in r_str:
                ranges = r_str[1:-1].split('-')
                output.append([int(ranges[0]), int(ranges[1])])
            else:
                print('Cannot parse hierarchy column')
    return output


def combine_hierarchies(h1, h2):
    combinations = [(l1, l2) for l1 in h1.columns for l2 in h2.columns]

    sizes = []
    combined = pd.DataFrame()

    for (l1, l2) in combinations:
        name = (l1, l2)
        combined[str(name)] = h1[l1].astype(str) + '::' + h2[l2].astype(str)
        groups = len(combined[str(name)].unique())
        sizes.append((str(name), groups))
    sizes.sort(reverse=True, key=lambda a: a[-1])
    cols = [size[0] for size in sizes]
    return combined[cols]


def draw_ml_error(x_real, y_real, y_pred):
    diffs = y_real != y_pred
    plt.scatter(x=x_real['Start Date num'], y=x_real['Time'], c=diffs)
    plt.show()


def get_hierarchy(name):
    base_path = os.path.join(os.path.dirname(__file__), '../output/hierarchies/NAME/output_ranges_hierarchy.csv')
    h_paths = [os.path.join(os.path.dirname(__file__), '../traditional_hierarchies/Departure Time.csv'),
               os.path.join(os.path.dirname(__file__), '../traditional_hierarchies/Duration.csv')]

    if name == 'traditional':
        h1 = pd.read_csv(h_paths[0], sep=';', decimal=',', header=None, dtype=str)
        h2 = pd.read_csv(h_paths[1], sep=';', decimal=',', header=None, dtype=str)
        return combine_hierarchies(h1, h2)
    else:
        path = base_path.replace('NAME', name)
        hierarchy = pd.read_csv(path, sep=';', decimal=',', header=None, dtype=str)
        return hierarchy


def calculate_distribution(original, labels, methods):
    df = pd.DataFrame(original)
    if len(df) != len(labels):
        df = df.T
    df['labels'] = labels

    grouped = df.groupby(by='labels')
    eqs = list(grouped)

    def calculate_col_distr(col):
        """calculates the real distribution"""
        distr = {}
        for qids, eq in eqs:
            col_counts = eq[eq.columns[col]].value_counts() / len(eq)
            distr[qids] = col_counts.to_dict()
        return distr

    def calculate_col_uniform_cat(col):
        """calculates uniform for categorical values"""
        distr = {}
        for qids, eq in eqs:
            uniques = eq[eq.columns[col]].unique()
            col_counts = {unique: 1 / len(uniques) for unique in uniques}
            distr[qids] = col_counts
        return distr

    def calculate_col_uniform_range(col):
        """assumes that label represents the range"""
        distr = {}
        for qids, eq in eqs:
            if qids != '*':
                splited = parse_column(qids)
                values = list(range(splited[col][0], splited[col][1] + 1))
            else:
                values = list(range(eq[eq.columns[col]].min(), eq[eq.columns[col]].max() + 1))

            perc = 1 / len(values)
            col_counts = {key: perc for key in values}
            distr[qids] = col_counts
        return distr

    def calculate_col_range_bounds(col):
        """assumes that label represents the range"""
        distr = {}
        for qids, eq in eqs:
            if qids != '*':
                splited = parse_column(qids)
                values = [(splited[col][0], splited[col][1])]
            else:
                values = [(eq[eq.columns[col]].min(), eq[eq.columns[col]].max())]

            perc = 1 / len(values)
            col_counts = {key: perc for key in values}
            distr[qids] = col_counts
        return distr

    def calculate_range_average(col):
        """assumes that label represents the range"""
        distr = {}
        for qids, eq in eqs:
            if qids != '*':
                splited = parse_column(qids)
                value = (splited[col][0] + splited[col][1]) / 2
            else:
                value = (eq[eq.columns[col]].min() + eq[eq.columns[col]].max()) / 2
            distr[qids] = {value: 1.0}
        return distr

    def calculate_data_average(col):
        """calculates the real numeric average"""
        distr = {}
        for qids, eq in eqs:
            value = eq[eq.columns[col]].mean()
            distr[qids] = {value: 1.0}
        return distr

    col_distr = {}
    methods_dict = {'distr': calculate_col_distr, 'range_uniform': calculate_col_uniform_range,
                    'cat_uniform': calculate_col_uniform_cat, 'range_avg': calculate_range_average,
                    'real_avg': calculate_data_average, 'range_bounds': calculate_col_range_bounds}

    for col_num, (col, method) in enumerate(zip(df.columns, methods)):
        col_distr[col] = methods_dict[method](col_num)

    return col_distr
