import os
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut

from utils import initialize


def create_hierarchies(data: pd.DataFrame, columns: List[str], cluster_amounts: List[int], multipliers: List[float],
                       generalize_functions, init_labels: ndarray = None) -> List[
    pd.DataFrame]:
    cluster_amounts.reverse()

    ml_columns = [f'{column} ml' for column in columns]
    # get data to cluster
    data_to_cluster = data[ml_columns]
    all_columns = columns + ml_columns

    # cluster in a bottom-up manner
    c_labels = agglomerative_clustering(data_to_cluster, cluster_amounts, init_labels)

    hierarchies = []
    for generalize in generalize_functions:
        #print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data[all_columns], columns, multipliers, c_labels))

    return hierarchies


def agglomerative_clustering(dataset, amounts, init_labels):
    uniques, reverse = np.unique(dataset.values, return_inverse=True, axis=0)
    model = AgglomerativeClustering(linkage=linkage, metric=metric, n_clusters=None, distance_threshold=0)
    model.fit(uniques)
    labels = [np.zeros(len(dataset),dtype=int)]

    for amount in amounts:
        c_labels = _hc_cut(amount, model.children_, model.n_leaves_).astype(int)[reverse]

        labels.append(c_labels)
    return labels


def output_centers(data, columns, multipliers, c_labels):
    data = data.copy()
    format_string = '{:.5f}'
    ml_columns = [f'{column} ml' for column in columns]

    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    data = data[ml_columns]
    data[ml_columns[0]] = ((data[ml_columns[0]] / multipliers[0]) + 0.5).astype(int)
    data[ml_columns[1]] = ((data[ml_columns[1]] / multipliers[1]) + 0.5).astype(int)

    for index, current_labels in enumerate(reversed(c_labels)):
        data['labels'] = current_labels

        centers = data.groupby(by=['labels']).mean()
        centers.columns = ['x_c', 'y_c']

        # string formatter
        centers = centers.applymap(format_string.format)

        merged = pd.merge(data, centers, how='left', left_on='labels', right_index=True)
        hierarchy[index + 1] = merged['x_c'] + '::' + merged['y_c']

    return hierarchy


# does not print a hierarchy but a level,label to distribution mapping. printing in hierarchy was already 1.6GB for the small range set
def output_distr(data, columns, multipliers, c_labels):
    data = data.copy()
    ml_columns = [f'{column} ml' for column in columns]

    distr = pd.DataFrame(dtype='<U32')

    data = data[ml_columns]
    data[ml_columns[0]] = ((data[ml_columns[0]] / multipliers[0]) + 0.5).astype(int)
    data[ml_columns[1]] = ((data[ml_columns[1]] / multipliers[1]) + 0.5).astype(int)

    for index, current_labels in enumerate(reversed(c_labels)):
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


def output_labels(data, columns, multipliers, c_labels):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    for index, current_labels in enumerate(reversed(c_labels)):
        hierarchy[index + 1] = current_labels

    return hierarchy


def output_ranges(data, columns, multipliers, c_labels):
    hierarchy = pd.DataFrame(dtype='<U32')
    hierarchy[0] = data[columns[0]].astype(str) + '::' + data[columns[1]].astype(str)

    # calculate range of all points
    df = pd.DataFrame(dtype=float)
    df['x'] = ((data[f'{columns[0]} ml'] / multipliers[0]) + 0.5).astype(int)
    df['y'] = ((data[f'{columns[1]} ml'] / multipliers[1]) + 0.5).astype(int)

    # calculate ranges for each level
    for index, current_labels in enumerate(reversed(c_labels)):
        df['labels'] = current_labels

        grouped = df.groupby(by=['labels'])
        maxs = grouped[['x', 'y']].max().astype(str)
        mins = grouped[['x', 'y']].min().astype(str)
        ranges_str = '[' + mins['x'] + '-' + maxs['x'] + ']::[' + mins['y'] + '-' + maxs['y'] + ']'
        merged = pd.merge(df, ranges_str.rename('output'), how='left', left_on='labels', right_index=True)

        hierarchy[index + 1] = merged['output']

    return hierarchy




if __name__ == '__main__':
    output_name = 'agglom'
    input_path = '../datasets/flight_fare.csv'
    output_path = f'../output/hierarchies/{output_name}/'

    columns = ['Departure Time num', 'Duration']
    weights = [1.0, 1.0]
    linkage = 'ward'
    metric = None

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

    # applies the weights
    input_df = initialize(original_input, columns, weights)
    # can be used to set a fixed initialization level (zie discussion in the paper)
    init_labels = None

    hierarchies = create_hierarchies(input_df, columns, node_amounts, weights, functions, init_labels=init_labels)

    for hierarchy, func in zip(hierarchies, functions):
        print(f'Writing hierarchy: {func.__name__}')
        hierarchy.to_csv(f'{output_path}{func.__name__}_hierarchy.csv', sep=';', decimal=',', index=False, header=False)
