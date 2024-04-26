import os
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray

import Mondrian_clustering as mc
from utils import initialize


def cell_ranges(data, labels: List[str], model: mc.MondrianClustering):
    hierarchy = pd.DataFrame(dtype='<U32')

    hierarchy[0] = data[labels[0]].astype(str) + '::' + data[labels[1]].astype(str)

    def range_to_string(current_range):
        x0, y0, x1, y1 = current_range
        return np.array([f'[{x0}-{x1}]::[{y0}-{y1}]'], dtype='<U32')

    for level in reversed(range(len(model.ranges))):
        c_labels = model.labels[level]
        ranges = model.ranges[level]
        ranges_str = np.apply_along_axis(range_to_string, 1, ranges)
        hierarchy[len(model.ranges) - level] = ranges_str[c_labels]
    return hierarchy


def output_ranges(data, columns: List[str], model: mc.MondrianClustering):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    # calculate range of all points
    ml_columns = [f'{column} ml' for column in columns]
    df = pd.DataFrame(dtype=int)
    df['x'], df['y'] = data[ml_columns[0]].astype(int), data[ml_columns[1]].astype(int)

    # calculate ranges for each level
    for index, current_labels in enumerate(reversed(model.labels)):
        df['labels'] = current_labels

        grouped = df.groupby(by=['labels'])
        maxs = grouped[['x', 'y']].max().astype(str)
        mins = grouped[['x', 'y']].min().astype(str)
        ranges_str = '[' + mins['x'] + '-' + maxs['x'] + ']::[' + mins['y'] + '-' + maxs['y'] + ']'
        merged = pd.merge(df, ranges_str.rename('output'), how='left', left_on='labels', right_index=True)

        hierarchy[index + 1] = merged['output']

    return hierarchy


def output_labels(data, columns: List[str], model: mc.MondrianClustering):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    for index, current_labels in enumerate(reversed(model.labels)):
        hierarchy[index + 1] = current_labels

    return hierarchy


# does not print a hierarchy but a level,label to distribution mapping. printing in hierarchy was already 1.6GB for the small range set
def output_distr(data, columns: List[str], model: mc.MondrianClustering):
    data = data.copy()
    ml_columns = [f'{column} ml' for column in columns]

    distr = pd.DataFrame(dtype='<U32')

    data = data[ml_columns].astype(int)

    for index, current_labels in enumerate(reversed(model.labels)):
        data['labels'] = current_labels

        grouped = data.groupby(by=['labels'])
        col_distr = []
        for col in ml_columns:
            col_distr.append(grouped[col].value_counts())

        level_distr = []
        for label in range(current_labels.max() + 1):
            level_distr.append([distr[label].to_dict() for distr in col_distr])
        distr[index] = pd.Series(level_distr)

    return distr


def create_hierarchies(data: pd.DataFrame, target: pd.Series, columns: List[str], cluster_amounts: List[int],
                       weights: List[float], dim_strategy: mc.DimSelectionStrategy, max_best: str,
                       generalize_functions, init_labels: ndarray = None) -> List[pd.DataFrame]:
    # get data to cluster
    data_to_cluster = data[[f'{column} ml' for column in columns]]

    # cluster using mondrian
    model = mc.MondrianClustering(cluster_amounts, dim_strategy=dim_strategy, max_best=max_best)
    model.fit_predict(data_to_cluster, target, weights=weights, init_labels=init_labels, progress=False)

    hierarchies = []
    for generalize in generalize_functions:
        # print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data, columns, model))

    return hierarchies

if __name__ == '__main__':

    output_name = 'uniformity_mondrian'
    input_path = '../datasets/flight_fare.csv'
    output_path = f'../output/hierarchies/{output_name}/'

    columns = ['Departure Time num', 'Duration']
    weights = [1.0, 1.0]

    # max possible for this data is 13549
    node_amounts = [7649, 7129, 6240, 5537, 4913, 4563, 3883, 3401, 3286, 2323, 2104, 2050, 2026, 1924, 1460, 1202,
                       1198, 1148, 920, 873, 866, 666, 665, 551, 488, 482, 481, 480, 423, 307, 290, 267, 261, 258, 257,
                       255, 168, 164, 158, 154, 141, 136, 93, 91, 86, 85, 81, 74, 54, 49, 47, 44, 29, 28, 26, 24, 15, 8,
                       4, 2]

    filtered = [node_amounts[0]]
    for i in node_amounts:
        if filtered[-1] * 0.95 > i:
            filtered.append(i)
    node_amounts = filtered

    # each function is a representation to print as hierarchy.
    # For our experiments we only printed the range representation and simulated the other representations.
    functions = [output_ranges]

    os.makedirs(output_path, exist_ok=True)
    original_input = pd.read_csv(input_path, sep=';', decimal=',')

    # Certain priority functions my try to take into account the ML target value
    # target = original_input['Price']
    # target = original_input['Price class']
    target = None

    input_df = initialize(original_input, columns)
    # can be used to set a fixed initialization level (zie discussion in the paper)
    init_labels = None

    # Multiple dim selection strategies where implemented.
    # Only best UNIFORMITY and max RELATIVE_RANGE results where used. As they were the most promising.
    hierarchies = create_hierarchies(input_df, target, columns, node_amounts, weights,
                                     mc.DimSelectionStrategy.UNIFORMITY, 'best', functions, init_labels=init_labels)

    for hierarchy, func in zip(hierarchies, functions):
        print(f'Writing hierarchy: {func.__name__}')
        hierarchy.to_csv(f'{output_path}{func.__name__}_hierarchy.csv', sep=';', decimal=',', index=False, header=False)
