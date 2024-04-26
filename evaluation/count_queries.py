import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import faulthandler

faulthandler.enable()


def get_random_queries(amount, domains):
    queries = []
    qid_values = [
        np.arange(domain[0], domain[1] + 1) if isinstance(domain, tuple) and len(domain) == 2 else np.array(domain) for
        domain in domains]

    for _ in range(amount):
        # random select the amount of dims
        dims = rng.integers(1, len(domains), endpoint=True)
        # random select dims
        selected_dims = rng.choice(len(domains), size=dims, replace=False, shuffle=False)

        qid_subsets = []
        for dim in selected_dims:
            # for each dim random select subset size
            subset_size = rng.integers(1, len(qid_values[dim]), endpoint=True)
            # for each dim take random subset
            qid_subsets.append(rng.choice(qid_values[dim], size=subset_size, replace=False, shuffle=False))
        queries.append((selected_dims, qid_subsets))
    return queries


def real_counts(data, queries):
    data_np = np.array(data.T)
    rows = len(data_np[0])
    counts = []

    for query in queries:
        filtered_data = data_np[query[0]]
        final_filter = np.ones(rows, dtype=bool)
        for qid in range(len(filtered_data)):
            final_filter &= np.isin(filtered_data[qid], query[1][qid])
        counts.append(final_filter.sum())

    return np.array(counts)


# START distr
eqs = []
distr = []


def init(_eqs, _distr):
    global eqs, distr
    eqs = _eqs
    distr = _distr


def evaluate(query):
    current_estimate = 0
    for eq_num, (qid_val, group) in enumerate(eqs):

        current_percentage = 1.0
        # calculate percentages for eq matches
        for i, dim in enumerate(query[0]):
            q_val = qid_val if isinstance(qid_val, str) else qid_val[dim] if dim < len(group.columns) else qid_val[
                dim - 1]
            qi_intersect = np.intersect1d(list(distr[qid_orig[dim]][q_val].keys()), query[1][i], assume_unique=True)

            current_percentage *= np.vectorize(distr[qid_orig[dim]][q_val].__getitem__, otypes=[float])(
                qi_intersect).sum()

            if current_percentage == 0:
                break
        else:
            current_estimate += current_percentage * len(group)
    return current_estimate


def estimated_counts_distr(originals, methods, queries, anonymized):
    # get distributions for each qid in originals using provided hierarchy_levels
    distr = {}
    for qid in list(anonymized.columns):
        hierarchy = anonymized[qid]
        qid = qid.split('::')

        original = originals[qid]
        method = [methods[q] for q in qid]
        distr.update(utils.calculate_distribution(original, hierarchy, method))

    # calculate anonymized eq
    by = list(anonymized.columns)
    by = by[0] if len(by) == 1 else by
    eqs = list(anonymized.groupby(by=by))

    if parallel:
        with Pool(processes=cores, initargs=(eqs, distr), initializer=init) as pool:
            counts = list(tqdm(pool.imap(evaluate, queries), total=len(queries)))
    else:
        init(eqs, distr)
        counts = []
        for query in tqdm(queries, total=len(queries)):
            counts.append(evaluate(query))
    return np.array(counts)


# END MULTICORE

def calculate_original(original_path, qid):
    original_df = pd.read_csv(original_path, sep=';', decimal=',')
    originals = original_df.loc[:, qid]

    domains = []
    for col in originals.columns:
        if originals[col].dtype == 'O':
            # replace with numerical values for speed
            qi_mapping = np.unique(originals[col].values, return_inverse=True)[1]
            originals[col] = qi_mapping
        domains.append((min(originals[col]), max(originals[col])))

    queries = get_random_queries(amount_queries, domains)
    real_results = real_counts(originals, queries)
    return real_results, queries, originals


def calculate_hierarchy(real_results, queries, originals):
    out_path = f'../output/queries/{test_name}/'
    os.makedirs(out_path, exist_ok=True)

    for name in names:
        print(name)
        out_file = f'{out_path}{name}_{estimate_name}.csv'
        output = pd.DataFrame()

        hierarchy = utils.get_hierarchy(name)
        qids = ['::'.join(qid_orig)]

        for level, level_name in enumerate(hierarchy.columns[1:-1]):
            print(f'calculating {level=}')

            anonymized = pd.DataFrame()
            anonymized[qids[0]] = hierarchy[level_name]

            # assume certain k-value and calculate metrics
            eqs = anonymized.groupby(by=qids[0])

            estimated_result = estimated_counts_distr(originals, methods[estimate_name],
                                                      queries, anonymized)

            abs_errors = np.divide(np.abs(real_results - estimated_result), real_results,
                                   out=np.full(real_results.shape, np.NAN, dtype=float), where=real_results != 0)
            avg_abs_error = np.nanmean(abs_errors)
            std_abs_error = np.nanstd(abs_errors)

            output.loc[f'avg abs', f'{level_name}'] = avg_abs_error
            output.loc[f'std abs', f'{level_name}'] = std_abs_error
            output.loc[f'original eqs', f'{level_name}'] = len(eqs)

        output.to_csv(out_file, sep=';', decimal=',')


original_path = '../datasets/flight_fare.csv'
qid_orig = ['Departure Time num', 'Duration']

rng = np.random.default_rng(420)
parallel = True
cores = 16
amount_queries = 10000
# name of the distribution to estimate, all results can be done on a hierarchy representing ranges.
estimate_name = 'set'
#estimate_name = 'real' # ==distribution
#estimate_name = 'uniform'

# name of the folder to put the results in
test_name = 'flight_fare'

# names of hierarchies to run
names = ['agglom', 'kmeans', 'nn', 'traditional', 'range_mondrian', 'uniformity_mondrian',
         'weighted_uniformity_mondrian']

methods = {'uniform': {'Departure Time num': 'range_uniform', 'Arrival Time num': 'range_uniform',
                       'Duration': 'range_uniform'},
           'real': {'Departure Time num': 'distr', 'Arrival Time num': 'distr', 'Duration': 'distr'},
           'set': {'Departure Time num': 'cat_uniform', 'Arrival Time num': 'cat_uniform', 'Duration': 'cat_uniform'}}

if __name__ == '__main__':
    print(estimate_name)
    print('calculating original')
    real_results, queries, originals = calculate_original(original_path, qid_orig)
    print('Calculating hierarchies')
    calculate_hierarchy(real_results, queries, originals)
