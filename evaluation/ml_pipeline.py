import os

os.environ['OMP_NUM_THREADS'] = '1'

from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBRegressor

from ..hierachy_creators import agglom_hierarchy as bu_hierarchy
from ..hierachy_creators import top_down_clustering_hierarchy as td_hierarchy
from ..hierachy_creators import mondrian_hierachy as mondrian
from tqdm import tqdm

from pre_processors import MeanPreprocessor, MinMaxRangePreprocessor, SetMeanPreprocessor, \
    DistrSamplePreprocessor, RangeMeanPreprocessor, SetSamplePreprocessor, RangeSamplePreprocessor
from utils import combine_hierarchies
from ..hierachy_creators.utils import initialize

import plotly.graph_objects as go


def generate_train_test():
    global config
    train_test_splits = config['train_test_splits']
    output_root = config['output_root']
    ml_data_path = config['ml_data_path']

    dataset_size = len(pd.read_csv(ml_data_path, sep=';', decimal=','))
    rng = np.random.default_rng()
    shuffled_index = np.arange(dataset_size, dtype=int)
    rng.shuffle(shuffled_index)
    splits = np.array_split(shuffled_index, train_test_splits)

    for i in tqdm(range(train_test_splits), total=train_test_splits, desc='Creating train/test splits'):
        out_dir = f'{output_root}/{i}/'
        os.makedirs(out_dir, exist_ok=True)
        pd.Series(splits[i], name='test_index').to_csv(f'{out_dir}test_index.csv', index=False)


def create_hierarchy(combo):
    index, algorithm, config, df = combo
    folder = f'{config["output_root"]}{index}/'
    test_index = pd.read_csv(f'{folder}test_index.csv', sep=';', decimal=',')
    training_set = df[~df.index.isin(test_index['test_index'])]
    target = training_set[config['target_col']]
    training_set = training_set[config['hierarchy_cols']]
    hierarchies_folder = f'{folder}hierarchies/'
    os.makedirs(hierarchies_folder, exist_ok=True)

    cluster_amounts = []
    if algorithm != traditional:
        # get cluster amounts
        traditional_hierarchy = pd.read_csv(f'{hierarchies_folder}traditional.csv', sep=';', decimal=',', header=None, dtype=str)
        cluster_amounts = traditional_hierarchy.nunique().values[1:-1].tolist()

        removed_levels = []
        filtered = [cluster_amounts[0]]
        for index, amount in enumerate(cluster_amounts[1:]):
            if filtered[-1] * 0.95 > amount:
                filtered.append(amount)
            else:
                removed_levels.append(index + 1)
        cluster_amounts = filtered

        removed = pd.Series(removed_levels, name='removed')
        removed.to_csv(f'{folder}removed_levels.csv', index=False)

    hierarchy = algorithm(training_set, target, cluster_amounts)
    hierarchy.to_csv(f'{hierarchies_folder}{algorithm.__name__}.csv', sep=';', decimal=',', index=False, header=False)


def create_hierarchies():
    global config
    train_test_splits = config['train_test_splits']
    ml_data_path = config['ml_data_path']
    algorithms = config['algorithms']
    cols = list(config['hierarchy_cols'])
    cols.append(config['target_col'])

    df = pd.read_csv(ml_data_path, sep=';', decimal=',')[cols]

    # als traditional methode moet gedraaid worden moet deze eerst zodat we de cluster hoeveelheden kunnen gebruiken
    if traditional in algorithms:
        jobs = [(i, traditional, config, df) for i in range(train_test_splits)]
        if not debug_mode:
            with Pool(processes=cores) as pool:
                list(tqdm(pool.imap(create_hierarchy, jobs), total=len(jobs), desc='Creating traditional hierarchies'))
        else:
            for job in tqdm(jobs, total=len(jobs), desc='Creating traditional hierarchies'):
                create_hierarchy(job)

    jobs = [(i, algorithm, config, df) for i in range(train_test_splits) for algorithm in algorithms if
            algorithm != traditional]
    if not debug_mode:
        with Pool(processes=cores) as pool:
            list(tqdm(pool.imap(create_hierarchy, jobs), total=len(jobs), desc='Creating hierarchies'))
    else:
        for job in tqdm(jobs, total=len(jobs), desc='Creating hierarchies'):
            create_hierarchy(job)


def traditional(df, target, cluster_amounts):
    h1 = pd.read_csv('../traditional_hierarchies/Departure Time.csv', sep=';', decimal=',', header=None, dtype=str)
    h2 = pd.read_csv('../traditional_hierarchies/Duration.csv', sep=';', decimal=',', header=None, dtype=str)

    combined = combine_hierarchies(h1, h2)
    combined.drop_duplicates(inplace=True)
    combined.set_index('(0, 0)', drop=False, inplace=True)

    columns = list(df.columns)
    df_combined = df[columns[0]].astype(str) + '::' + df[columns[1]].astype(str)
    hierarchy = combined.loc[df_combined]
    return hierarchy


def agglom(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns), multipliers)

    bu_hierarchy.linkage = 'ward'
    bu_hierarchy.metric = None

    hierarchy = bu_hierarchy.create_hierarchies(data, list(df.columns), cluster_amounts, multipliers,
                                                [bu_hierarchy.output_ranges])
    return hierarchy[0]


def agglom2(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns), multipliers)

    bu_hierarchy.linkage = 'complete'
    bu_hierarchy.metric = 'manhattan'

    hierarchy = bu_hierarchy.create_hierarchies(data, list(df.columns), cluster_amounts, multipliers,
                                                [bu_hierarchy.output_ranges])
    return hierarchy[0]


def nn(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns), multipliers)

    td_hierarchy.cluster_algorithm = td_hierarchy.apply_nn_constrained

    hierarchy = td_hierarchy.create_hierarchies(data, list(df.columns), cluster_amounts, multipliers,
                                                [td_hierarchy.output_ranges])
    return hierarchy[0]


def kmeans(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns), multipliers)

    td_hierarchy.cluster_algorithm = td_hierarchy.apply_kmeans

    hierarchy = td_hierarchy.create_hierarchies(data, list(df.columns), cluster_amounts, multipliers,
                                                [td_hierarchy.output_ranges])
    return hierarchy[0]


def constrained_kmeans(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns), multipliers)

    td_hierarchy.cluster_algorithm = td_hierarchy.apply_kmeans_constrained

    hierarchy = td_hierarchy.create_hierarchies(data, list(df.columns), cluster_amounts, multipliers,
                                                [td_hierarchy.output_ranges])
    return hierarchy[0]


def range_mondrian(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns))

    hierarchy = mondrian.create_hierarchies(data, target, list(df.columns), cluster_amounts, multipliers,
                                            mondrian.mc.DimSelectionStrategy.RELATIVE_RANGE, 'max',
                                            [mondrian.output_ranges])
    return hierarchy[0]


def weighted_range_mondrian(df, target, cluster_amounts):
    multipliers = [30.0, 1.0]

    data = initialize(df, list(df.columns))

    hierarchy = mondrian.create_hierarchies(data, target, list(df.columns), cluster_amounts, multipliers,
                                            mondrian.mc.DimSelectionStrategy.RELATIVE_RANGE, 'max',
                                            [mondrian.output_ranges])
    return hierarchy[0]


def weighted2_range_mondrian(df, target, cluster_amounts):
    multipliers = [1.0, 30.0]

    data = initialize(df, list(df.columns))

    hierarchy = mondrian.create_hierarchies(data, target, list(df.columns), cluster_amounts, multipliers,
                                            mondrian.mc.DimSelectionStrategy.RELATIVE_RANGE, 'max',
                                            [mondrian.output_ranges])
    return hierarchy[0]


def uniformity_mondrian(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns))

    hierarchy = mondrian.create_hierarchies(data, target, list(df.columns), cluster_amounts, multipliers,
                                            mondrian.mc.DimSelectionStrategy.UNIFORMITY, 'best',
                                            [mondrian.output_ranges])
    return hierarchy[0]


def mse_mondrian(df, target, cluster_amounts):
    multipliers = [1.0, 1.0]

    data = initialize(df, list(df.columns))

    hierarchy = mondrian.create_hierarchies(data, target, list(df.columns), cluster_amounts, multipliers,
                                            mondrian.mc.DimSelectionStrategy.MSE, 'best',
                                            [mondrian.output_ranges])
    return hierarchy[0]


def do_ml(combo):
    df, index, config, preprocessor, hierarchy_cols, cores = combo
    # create training test split
    folder = f'{config["output_root"]}{index}/'
    test_index = pd.read_csv(f'{folder}test_index.csv', sep=';', decimal=',')
    target = config['target_col']
    train_set = df[~df.index.isin(test_index['test_index'])]
    test_set = df[df.index.isin(test_index['test_index'])]
    train_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)

    train_x, train_y, test_x, test_y = train_set.drop(columns=target), train_set[target], test_set.drop(columns=target), \
        test_set[target]

    del df, train_set, test_set

    # filter numeric/categorical not being hierarchy cols
    numerical, categorical = [], []
    for col in train_x.columns:
        if col not in hierarchy_cols:
            if is_numeric_dtype(train_x[col]) and train_x[col].nunique() > 2:
                numerical.append(col)
            else:
                # categorical should be one hot encoded
                categorical.append(col)

    # create pipeline
    pipe = Pipeline([('transformer', ColumnTransformer(
        [('categorical', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical),
         ('numerical', MinMaxScaler(feature_range=(0, 1)), numerical),
         ('hierarchy_cols', preprocessor, hierarchy_cols)], n_jobs=cores, verbose_feature_names_out=False,
        sparse_threshold=0)), ('model', XGBRegressor(n_jobs=cores))])
    # add tree_method='gpu_hist', gpu_id=0,  to XGBRegressor to improve runtime when a gpu and cuda is installed

    pipe.fit(train_x, train_y)
    y_pred = pipe.predict(test_x)
    r2 = r2_score(test_y, y_pred)
    rmse = mean_squared_error(test_y, y_pred, squared=False)
    scores = {'r2': r2, 'RMSE': rmse}
    return scores


def ml_orig_base():
    global config

    def print_result(result, index, name):
        # print results
        output_root = config['output_root']
        out_path = f'{output_root}{index}/results/'
        os.makedirs(out_path, exist_ok=True)
        out_file = f'{out_path}{name}.csv'
        out_series = pd.Series(name=0)
        for key, value in result.items():
            out_series[key] = value
        out_series.to_csv(out_file, sep=';', decimal=',')

    all_cols = config['ml_columns'] + config['hierarchy_cols'] + [config['target_col']]
    ml_data_path = config['ml_data_path']
    hierarchy_cols = config['hierarchy_cols']
    train_test_splits = config['train_test_splits']

    df = pd.read_csv(ml_data_path, sep=';', decimal=',')[all_cols]

    parallel_jobs = 4 if not debug_mode else 1
    cores_in_job = cores // parallel_jobs

    # add original jobs
    jobs = [(df, index, config, 'drop', [], cores_in_job) for index in range(train_test_splits)]

    # add baseline jobs
    jobs.extend(
        [(df, index, config, 'drop', hierarchy_cols, cores_in_job) for index in range(train_test_splits)])

    names = ['original', 'baseline', 'x_dropped', 'y_dropped']

    # add x-dropped jobs
    jobs.extend(
        [(df, index, config, 'drop', [hierarchy_cols[0]], cores_in_job) for index in range(train_test_splits)])
    # add y-dropped jobs
    jobs.extend(
        [(df, index, config, 'drop', [hierarchy_cols[1]], cores_in_job) for index in range(train_test_splits)])

    # run jobs
    if not debug_mode:
        with Pool(processes=parallel_jobs) as pool:
            index = 0
            name_index = 0
            for result in tqdm(pool.imap(do_ml, jobs), total=len(jobs), desc='ML original/baseline'):
                print_result(result, index, names[name_index])
                index = (index + 1) % train_test_splits
                if index == 0:
                    name_index += 1
    else:
        index = 0
        name_index = 0
        for job in tqdm(jobs, total=len(jobs), desc='ML original/baseline'):
            result = do_ml(job)
            print_result(result, index, names[name_index])
            index = (index + 1) % train_test_splits
            if index == 0:
                name_index += 1


def ml_hierarchies():
    global config

    all_cols = config['ml_columns'] + config['hierarchy_cols'] + [config['target_col']]
    ml_data_path = config['ml_data_path']
    hierarchy_cols = config['hierarchy_cols']
    train_test_splits = config['train_test_splits']
    output_root = config['output_root']

    df = pd.read_csv(ml_data_path, sep=';', decimal=',')[all_cols]

    parallel_jobs = cores
    cores_in_job = 1

    # for each hierarchy type
    total_hierarchies = len(config['algorithms']) * len(config['preprocessors']) * train_test_splits
    pbar = tqdm(total=total_hierarchies, desc='ML runs')

    for hierarchy_type in config['algorithms']:
        # for each train_test_split
        for index in range(train_test_splits):
            hierarchy_pad = f'{output_root}{index}/hierarchies/{hierarchy_type.__name__}.csv'
            hierarchy = pd.read_csv(hierarchy_pad, sep=';', decimal=',', header=None, dtype=str)
            # for each preprocessor
            for preprocessor in config['preprocessors']:
                # for each hierarchy level (not first and last)
                jobs = [(df, index, config, preprocessor(hierarchy[level]), hierarchy_cols, cores_in_job) for level in
                        hierarchy.columns[1:-1]]
                results = []
                if not debug_mode:
                    with Pool(processes=parallel_jobs) as pool:
                        results = list(tqdm(pool.imap(do_ml, jobs), total=len(jobs),
                                            desc=f'ML {hierarchy_type.__name__} {jobs[0][3].name}'))
                else:
                    for job in tqdm(jobs, total=len(jobs), desc=f'ML {hierarchy_type.__name__} {jobs[0][3].name}'):
                        results.append(do_ml(job))

                # save for each combo in between
                out_path = f'{output_root}{index}/results/{hierarchy_type.__name__}/'
                os.makedirs(out_path, exist_ok=True)
                out_file = f'{out_path}{jobs[0][3].name}.csv'

                out_df = pd.DataFrame(dtype=object)
                for level, result in zip(list(hierarchy.columns[1:-1]), results):
                    out_series = pd.Series(name=level)
                    for key, value in result.items():
                        out_series[key] = value
                    out_series['eqs'] = hierarchy[level].nunique()
                    out_df[level] = out_series
                out_df.to_csv(out_file, sep=';', decimal=',')
                pbar.update(1)
    pbar.close()


def calc_draw_data(job):
    algorithm, preprocessor, config = job

    output_root = config['output_root']
    train_test_splits = config['train_test_splits']

    y_values = []
    x_values = []
    samples_in_level = [0] * config['max_levels']
    for sample in range(train_test_splits):
        y_values.append([0] * config['max_levels'])
        x_values.append([0] * config['max_levels'])
        skipped_levels = pd.read_csv(f'{output_root}{sample}/removed_levels.csv').values[:, 0].tolist()
        data = pd.read_csv(f'{output_root}{sample}/results/{algorithm.__name__}/{preprocessor.name}.csv', sep=';',
                           decimal=',')
        data.set_index(data.columns[0], inplace=True)

        real_level = 0
        for level in range(config['max_levels']):
            if algorithm.__name__ == 'traditional' or level not in skipped_levels:
                y_values[sample][level] = data.loc[config['draw_metric'], data.columns[real_level]]
                x_values[sample][level] = data.loc['eqs', data.columns[real_level]]
                samples_in_level[level] += 1
                real_level += 1

    # aggregate the samples
    y_averages = []
    y_std = []
    x_averages = []
    for level in range(len(samples_in_level)):
        if samples_in_level[level] == 5:
            x_s = []
            y_s = []
            for sample in range(train_test_splits):
                x_s.append(x_values[sample][level])
                y_s.append(y_values[sample][level])
            y_averages.append(np.average(y_s))
            x_averages.append(np.average(x_s))
            y_std.append(np.std(y_s))

    return (f'{algorithm.__name__}__{preprocessor.name}', y_averages, y_std, x_averages)


def draw_plotly(data_to_draw):
    global config
    plots = []

    for name, y_averages, y_std, x_averages in data_to_draw:
        c = list(np.random.choice(range(256), size=3))
        plots.append(go.Scatter(x=x_averages, y=y_averages, mode='lines', line={'color': f'rgb({c[0]},{c[1]},{c[2]})'},
                                name=name, legendgroup=name))
        y_upper = np.array(y_averages) + np.array(y_std)
        y_lower = np.array(y_averages) - np.array(y_std)
        if draw_std_div:
            plots.append(go.Scatter(
                x=x_averages + x_averages[::-1],  # x, then x reversed
                y=y_upper.tolist() + y_lower[::-1].tolist(),  # upper, then lower reversed
                fill='toself',
                fillcolor=f'rgba({c[0]},{c[1]},{c[2]},0.2)',
                line={'color': 'rgba(255,255,255,0)'},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=name))

    fig = go.Figure(data=plots)
    fig.update_layout(
        xaxis={'autorange': 'reversed'},
        xaxis_type='log'
    )
    fig.write_html(f'{config["output_root"]}drawing_{config["draw_metric"]}.html')


def draw_results():
    global config
    output_root = config['output_root']
    algorithms = config['algorithms']
    preprocessors = config['preprocessors']
    train_test_splits = config['train_test_splits']

    # list containing name, list of y values average, list of y values std div, list of x values
    data_to_plot = []

    jobs = [(algorihm, preprocessor, config) for algorihm in algorithms for preprocessor in
            preprocessors]

    if debug_mode:
        for job in tqdm(jobs, total=len(jobs), desc='Drawing preprocessing'):
            data_to_plot.append(calc_draw_data(job))
    else:
        with Pool(processes=cores) as pool:
            data_to_plot.extend(tqdm(pool.imap(calc_draw_data, jobs), total=len(jobs), desc='Drawing preprocessing'))

    # add original
    max_x = max([max(p_data[3]) for p_data in data_to_plot])
    min_x = min([min(p_data[3]) for p_data in data_to_plot])
    original_y_s = []
    for sample in range(train_test_splits):
        original_df = pd.read_csv(f'{output_root}{sample}/results/original.csv', sep=';', decimal=',')
        original_df.set_index(original_df.columns[0], inplace=True)
        original_y_s.append(original_df.loc[config['draw_metric'], original_df.columns[0]])
    original_y_avg = np.average(original_y_s)
    original_y_std = np.std(original_y_s)
    data_to_plot.append(('original', [original_y_avg] * 2, [original_y_std] * 2, [max_x, min_x]))

    # add baseline
    baseline_y_s = []
    for sample in range(train_test_splits):
        baseline_df = pd.read_csv(f'{output_root}{sample}/results/baseline.csv', sep=';', decimal=',')
        baseline_df.set_index(baseline_df.columns[0], inplace=True)
        baseline_y_s.append(baseline_df.loc[config['draw_metric'], baseline_df.columns[0]])
    baseline_y_avg = np.average(baseline_y_s)
    baseline_y_std = np.std(baseline_y_s)
    data_to_plot.append(('baseline', [baseline_y_avg] * 2, [baseline_y_std] * 2, [max_x, min_x]))

    # add x_dropped
    x_dropped_y_s = []
    for sample in range(train_test_splits):
        x_dropped_df = pd.read_csv(f'{output_root}{sample}/results/x_dropped.csv', sep=';', decimal=',')
        x_dropped_df.set_index(x_dropped_df.columns[0], inplace=True)
        x_dropped_y_s.append(x_dropped_df.loc[config['draw_metric'], x_dropped_df.columns[0]])
    x_dropped_y_avg = np.average(x_dropped_y_s)
    x_dropped_y_std = np.std(x_dropped_y_s)
    data_to_plot.append(('x_dropped', [x_dropped_y_avg] * 2, [x_dropped_y_std] * 2, [max_x, min_x]))

    # add y_dropped
    y_dropped_y_s = []
    for sample in range(train_test_splits):
        y_dropped_df = pd.read_csv(f'{output_root}{sample}/results/y_dropped.csv', sep=';', decimal=',')
        y_dropped_df.set_index(y_dropped_df.columns[0], inplace=True)
        y_dropped_y_s.append(y_dropped_df.loc[config['draw_metric'], y_dropped_df.columns[0]])
    y_dropped_y_avg = np.average(y_dropped_y_s)
    y_dropped_y_std = np.std(y_dropped_y_s)
    data_to_plot.append(('y_dropped', [y_dropped_y_avg] * 2, [y_dropped_y_std] * 2, [max_x, min_x]))

    draw_plotly(data_to_plot)


if __name__ == '__main__':
    debug_mode = False  # disables parallel runs
    draw_std_div = False
    cores = 8
    config = {'ml_data_path': '../datasets/flight_fare.csv',
              'output_root': f'../output/ml/', 'train_test_splits': 5,
              'hierarchy_cols': ['Departure Time num', 'Duration'],
              'target_col': 'Price',
              'ml_columns': ['Total Stops', 'Days left', 'Airline', 'class', 'Departure location', 'Arrival location'],
              'algorithms': [traditional, agglom, nn, kmeans, range_mondrian, uniformity_mondrian, weighted_range_mondrian,
                             weighted2_range_mondrian],
              'preprocessors': [MeanPreprocessor, MinMaxRangePreprocessor, RangeMeanPreprocessor, SetSamplePreprocessor,
                                SetMeanPreprocessor, DistrSamplePreprocessor, RangeSamplePreprocessor],
              'draw_metric': 'RMSE',
              'max_levels': 70 # max level to draw
              }
    # stages to run, you can disable result drawing
    stages = [generate_train_test,
        create_hierarchies,
        ml_orig_base,
        ml_hierarchies,
        draw_results
    ]

    for fn in tqdm(stages, total=len(stages), desc="Stage"):
        fn()
