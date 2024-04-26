import os

os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree

from multiprocessing import Pool
from typing import List, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from numpy import ndarray
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from tqdm import tqdm

from utils import initialize


def summed_distances_fast(mat1, mat2):
    unique1, inverse1, counts1 = np.unique(mat1, return_inverse=True, return_counts=True, axis=0)
    unique2, inverse2, counts2 = np.unique(mat2, return_inverse=True, return_counts=True, axis=0)
    
    distances = pairwise_distances(unique1, unique2, metric='euclidean')
    summed_distances = np.sum(np.multiply(distances, counts2), axis=1)
    return summed_distances[inverse1]


def nnit(mat, clsize=10, method='random'):
    clsize = np.ceil(np.arange(1, mat.shape[0] + 1) / (mat.shape[0] / clsize)).astype(int)
    clsize = np.bincount(clsize)
    lab = np.full(mat.shape[0], np.nan, dtype=float)

    # init sum of distances
    if method == 'maxd' or method == 'mind':
        distance_sums = summed_distances_fast(mat, mat)

    cpt = 0
    while np.isnan(lab).sum() > 0:
        lab_ii = np.where(np.isnan(lab))[0]
        if method == 'random':
            ii = np.random.choice(len(lab_ii), 1)[0]
        elif method == 'maxd':
            ii = np.argmax(distance_sums)
        elif method == 'mind':
            ii = np.argmin(distance_sums)
        else:
            raise ValueError('unknown method')

        lab_m = np.full(lab_ii.shape, np.nan, dtype=float)

        # calculate clsize[cpt] nearest neighbors of mat[ii] to mat[lab_ii] using kdtree
        tree = KDTree(mat[lab_ii])
        indishes = tree.query(mat[ii].reshape(1, 2), k=clsize[cpt+1], return_distance=False)[0]

        lab_m[indishes] = cpt
        lab[lab_ii] = lab_m

        if method == 'maxd' or method == 'mind':
            # remove indices rows
            distance_sums = np.delete(distance_sums, indishes)
            # remove distance of leftover to all indices points
            if len(distance_sums) != 0:
                to_remove = summed_distances_fast(mat[np.delete(lab_ii, indishes)], mat[lab_ii[indishes]])
                distance_sums -= to_remove
        cpt += 1
    if np.isnan(lab).sum() > 0:
        lab[np.where(np.isnan(lab))[0]] = cpt
    return lab.astype(int)


def create_hierarchies(data: pd.DataFrame, columns: List[str], cluster_amounts: List[int], multipliers: List[float],
                       generalize_functions, init_labels: ndarray = None) -> List[
    pd.DataFrame]:
    cluster_amounts.reverse()

    ml_columns = [f'{column} ml' for column in columns]
    # get data to cluster
    data_to_cluster = data[ml_columns]
    all_columns = columns + ml_columns

    # cluster in a top-down manner
    c_labels = top_down_clustering(data_to_cluster, cluster_amounts, init_labels)

    hierarchies = []
    for generalize in generalize_functions:
        #print(f'Creating hierarchy: {generalize.__name__}')
        hierarchies.append(generalize(data[all_columns], columns, multipliers, c_labels))

    return hierarchies


def apply_kmeans(dataset, n_clusters):
    mbk = MiniBatchKMeans(n_clusters=n_clusters, n_init=5, reassignment_ratio=0)
    labels = mbk.fit_predict(dataset)

    # removes skipped labels when empty cluster are created
    return np.unique(labels, return_inverse=True)[1]


def apply_xmeans(dataset, n_clusters):
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(dataset, amount_initial_centers).initialize()
    xmeans_instance = xmeans(dataset, initial_centers, n_clusters)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    labels = np.zeros(len(dataset), dtype=int)

    for cluster_label, cluster in enumerate(clusters):
        for row in cluster:
            labels[row] = cluster_label
    return labels


def get_min_cluster_size(dataset, n_clusters):
    # calculates size for equal distribution, it can occure that this causes a large set of identical points to be split
    # this is not allowed, so we recalculate the size taking this into account
    counts = np.sort(np.unique(dataset, axis=0, return_counts=True)[1])[::-1]
    data_size = len(dataset)
    min_size = int(data_size / n_clusters)
    remove_candidate = 0

    while min_size < counts[remove_candidate]:
        data_size -= counts[remove_candidate]
        n_clusters -= 1
        min_size = int(data_size / n_clusters)
        remove_candidate += 1
    return min_size


def apply_kmeans_constrained(dataset, n_clusters):
    size_min = get_min_cluster_size(dataset, n_clusters)
    # size_min = int(len(dataset) / n_clusters)
    mbk = KMeansConstrained(n_clusters=n_clusters, n_init=5, size_min=size_min)
    labels = mbk.fit_predict(dataset)

    # this algorithm can assign multiple equal points to different labels
    row_to_label_counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(dataset.tolist(), labels):
        row_to_label_counts[str(a)][b] += 1

    # smart assign grote groep van gelijke punten aan kleinste cluster
    row_to_label = {}
    current_counts = defaultdict(int)
    leftover_groups = {}
    for row, clusters in row_to_label_counts.items():
        if len(clusters.keys()) == 1:
            key, value = list(clusters.items())[0]
            current_counts[key] += value
            row_to_label[row] = key
        else:
            clusters['total'] = sum(clusters.values())
            leftover_groups[row] = clusters

    for row, clusters in sorted(leftover_groups.items(), key=lambda x: x[1]['total'], reverse=True):
        possible_labels = list(clusters.keys())
        possible_labels.remove('total')
        new_label = min(possible_labels, key=current_counts.__getitem__)

        row_to_label[row] = new_label
        current_counts[new_label] += clusters['total']

    df = pd.DataFrame(dataset)
    labels = df.apply(lambda row: row_to_label[str(row.tolist())], axis=1).values

    return labels


def apply_nn_constrained(dataset, n_clusters):
    labels = nnit(dataset, n_clusters, 'maxd')

    # this algorithm can assign multiple equal points to different labels
    row_to_label_counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(dataset.tolist(), labels):
        row_to_label_counts[str(a)][b] += 1

    # smart assign grote groep van gelijke punten aan kleinste cluster
    row_to_label = {}
    current_counts = defaultdict(int)
    leftover_groups = {}
    for row, clusters in row_to_label_counts.items():
        if len(clusters.keys()) == 1:
            key, value = list(clusters.items())[0]
            current_counts[key] += value
            row_to_label[row] = key
        else:
            clusters['total'] = sum(clusters.values())
            leftover_groups[row] = clusters

    for row, clusters in sorted(leftover_groups.items(), key=lambda x: x[1]['total'], reverse=True):
        possible_labels = list(clusters.keys())
        possible_labels.remove('total')
        new_label = min(possible_labels, key=current_counts.__getitem__)

        row_to_label[row] = new_label
        current_counts[new_label] += clusters['total']

    df = pd.DataFrame(dataset)
    labels = df.apply(lambda row: row_to_label[str(row.tolist())], axis=1).values

    return labels


def apply_agglom(dataset, n_clusters):
    uniques, reverse = np.unique(dataset, return_inverse=True, axis=0)

    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    labels = model.fit_predict(uniques)[reverse]
    # removes skipped labels when empty cluster are created
    return np.unique(labels, return_inverse=True)[1]


def get_new_cluster_amounts(np_dataset, prev_labels, needed_clusters):
    """Calculates for each previous cluster how many sub-clusters are needed,
     based on their current size in comparison to the total size"""

    # calculate cluster sizes
    c_sizes = np.bincount(prev_labels)
    total_records = len(np_dataset)

    # calculate num_unique for each cluster
    uniques = np.zeros(len(c_sizes), dtype=int)
    for label in range(len(c_sizes)):
        uniques[label] = len(np.unique(np_dataset[prev_labels == label], axis=0))

    # a prev cluster can't be devided into more clusters than unique values it has
    merged = np.vstack((c_sizes, uniques)).T
    cant_split_more = set()
    sub_clusters = np.apply_along_axis(lambda row: min(max(int((row[0] / total_records) * needed_clusters), 1), row[1]),
                                       1, merged)
    cant_split_more.update(np.nonzero(np.equal(uniques, sub_clusters))[0])

    # more groups will be needed, split current largest subgroups more
    new_cluster_sizes = np.divide(c_sizes, sub_clusters)

    all_labels = np.arange(len(c_sizes))
    while np.sum(sub_clusters) < needed_clusters:
        allowed_labels = np.where(~np.isin(all_labels, list(cant_split_more)))[0]
        label = allowed_labels[new_cluster_sizes[allowed_labels].argmax()]
        sub_clusters[label] += 1

        new_cluster_sizes[label] = c_sizes[label] / sub_clusters[label]

        if len(cant_split_more) == len(c_sizes):
            print('Cannot split further', needed_clusters)
            break

        if sub_clusters[label] == uniques[label]:
            cant_split_more.add(label)

    return sub_clusters


def recluster(input_date):
    amount, data = input_date
    if amount == 0:
        print('A subcluster amount of 0 occurred')
    elif amount == 1:
        return np.zeros(len(data), dtype=int)
    else:
        return cluster_algorithm(data, amount)


def recluster_parallel(dataset_np, new_cluster_amounts, prev_labels):
    c_labels = np.full(len(dataset_np), -1)

    dataset_df = pd.DataFrame(dataset_np)
    dataset_df['prev_labels'] = prev_labels
    grouped = dataset_df.groupby(by='prev_labels')[list(dataset_df.columns[:-1])]

    jobs = [(amount, grouped.get_group(label[0]).values) for label, amount in np.ndenumerate(new_cluster_amounts)]
    results = []

    if parallel:
        with Pool(processes=cores) as pool:
            max_ = np.sum(new_cluster_amounts)
            with tqdm(total=max_) as pbar:
                for result in pool.imap(recluster, jobs):
                    results.append(result)
                    pbar.update(len(np.unique(result)))
    else:
        for job in jobs:
            results.append(recluster(job))

    max_label = 0
    for label, new_amount in np.ndenumerate(new_cluster_amounts):
        label = label[0]
        data_filter = prev_labels == label
        current_labels = results[label]

        if (a := len(set(current_labels))) != new_cluster_amounts[label]:
            print('Cluster amount missmatch. Got: ', a, 'expected: ', new_cluster_amounts[label])

        current_max_label = np.max(current_labels)
        current_labels += max_label
        max_label += current_max_label + 1

        c_labels[data_filter] = current_labels
    return c_labels


def top_down_clustering(dataset, cluster_amounts, init_labels: List[int] = None) -> list[
    ndarray | Any]:
    if init_labels is not None and (init_label_amount := len(set(init_labels))) >= np.min(cluster_amounts):
        raise ValueError(
            f'There are more unique initial labels than the smallest cluster. Expected <{np.min(cluster_amounts)} got {init_label_amount}')

    labels = []
    dataset_np = dataset.values

    if init_labels is not None:
        # set the initial labels as the first hierachy_creators level
        labels.append(np.array(init_labels))
    else:
        # first level just assign every point label 0 (suppression level)
        labels.append(np.zeros(len(dataset), dtype=int))

    # all other clusters, apply hierachy_creators on each previous level cluster
    for amount in cluster_amounts:
        # calculate for each cluster, how many sub-clusters need to be generated
        new_cluster_amounts = get_new_cluster_amounts(dataset_np, labels[-1], amount)

        c_labels = recluster_parallel(dataset_np, new_cluster_amounts, labels[-1])
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


# The paper uses kmeans and nn, others also work but results where less interesting
# cluster_algorithm = apply_kmeans_constrained
cluster_algorithm = apply_kmeans
# cluster_algorithm = apply_xmeans
# cluster_algorithm = apply_agglom
linkage = 'ward'
#cluster_algorithm = apply_nn_constrained

cores = 8
parallel = True
if __name__ == '__main__':

    output_name = 'kmeans'
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

    print(output_name)
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
