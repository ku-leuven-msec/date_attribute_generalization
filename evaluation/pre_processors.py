from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


def parse_column(row):
    if isinstance(row, str):
        ranges_str = row.split('::')
    else:
        ranges_str = row
    output = []
    for r_str in ranges_str:
        try:
            val = int(r_str)
            output.extend([val, val])
        except ValueError:
            if '[' in r_str and ']' in r_str:
                ranges = r_str[1:-1].split('-')
                output.extend([int(ranges[0]), int(ranges[1])])
            else:
                print('Cannot parse hierarchy column')
    return np.array(output)


class HierarchyPreprocessorBase(BaseEstimator, TransformerMixin, ABC):

    def __init__(self, generalization, name):
        self.generalization = generalization
        self.just_fitted = False
        self.name = name

    @abstractmethod
    def train_transform(self, X, y=None):
        ...

    @abstractmethod
    def test_transform(self, X, y=None):
        ...

    def fit(self, X, y=None):
        self.just_fitted = True
        return self

    def transform(self, X, y=None):
        if self.just_fitted:
            self.just_fitted = False
            return self.train_transform(X, y)
        else:
            return self.test_transform(X, y)


class MeanPreprocessor(HierarchyPreprocessorBase):
    name = 'Mean'

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'Mean')

    def train_transform(self, X, y=None):
        X['generalization'] = self.generalization
        output = X.groupby(by='generalization').transform('mean')
        output = self.post_transformer.fit_transform(output)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)


class MinMaxRangePreprocessor(HierarchyPreprocessorBase):
    name = 'MinMax'

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'MinMax')

    @staticmethod
    def __np_to_df(X, np_array):
        tmp = pd.DataFrame(np_array)
        columns = []
        for name in X.columns:
            columns.extend([f'{name} min', f'{name} max'])
        tmp.columns = columns
        return tmp

    def train_transform(self, X, y=None):
        uniques, reverse = np.unique(self.generalization, return_inverse=True)
        tmp = np.vectorize(parse_column, signature='()->(n)')(uniques)
        tmp = tmp[reverse]
        tmp = self.__np_to_df(X, tmp)

        output = self.post_transformer.fit_transform(tmp)
        return output

    def test_transform(self, X, y=None):
        uniques, reverse = np.unique(X, return_inverse=True, axis=0)
        tmp = np.vectorize(parse_column, signature='(n)->(m)')(uniques)
        tmp = tmp[reverse]
        tmp = self.__np_to_df(X, tmp)

        output = self.post_transformer.transform(tmp)
        return output


class RangeMeanPreprocessor(HierarchyPreprocessorBase):
    name = 'RangeMean'

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'RangeMean')

    def train_transform(self, X, y=None):
        uniques, reverse = np.unique(self.generalization, return_inverse=True)
        tmp = np.vectorize(parse_column, signature='()->(n)')(uniques)

        means = np.zeros(shape=(tmp.shape[0], int(tmp.shape[1] / 2)), dtype=float)
        for i in range(len(X.columns)):
            means[:,i] = (tmp[:,i * 2] + tmp[:,i * 2 + 1]) / 2
        means = means[reverse]

        tmp = pd.DataFrame(means, columns=X.columns)

        output = self.post_transformer.fit_transform(tmp)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)


class RangeSamplePreprocessor(HierarchyPreprocessorBase):
    name = 'RangeSample'

    @staticmethod
    def __take_sample(column, column_names):
        ranges = parse_column(column.index[0])
        col_num = column_names.index(column.name)
        rng = np.random.default_rng()
        min_val = ranges[col_num*2]
        max_val = ranges[col_num*2+1]
        return rng.choice(np.arange(min_val,max_val+1), len(column))

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'RangeSample')

    def train_transform(self, X, y=None):
        column_names = list(X.columns)
        X['generalization'] = self.generalization
        output = X.set_index('generalization').groupby(level='generalization').transform(self.__take_sample, column_names).reset_index(drop=True)
        output = self.post_transformer.fit_transform(output)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)


class SetSamplePreprocessor(HierarchyPreprocessorBase):
    name = 'SetSample'

    @staticmethod
    def __take_sample(column):
        uniques = np.unique(column)
        rng = np.random.default_rng()
        return rng.choice(uniques, len(column))

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'SetSample')

    def train_transform(self, X, y=None):
        X['generalization'] = self.generalization
        output = X.groupby(by='generalization').transform(self.__take_sample)
        output = self.post_transformer.fit_transform(output)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)


class SetMeanPreprocessor(HierarchyPreprocessorBase):
    name = 'SetMean'

    @staticmethod
    def __take_sample(column):
        uniques = np.unique(column)
        mean = np.mean(uniques)
        return np.full(len(column), mean)

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'SetMean')

    def train_transform(self, X, y=None):
        X['generalization'] = self.generalization
        output = X.groupby(by='generalization').transform(self.__take_sample)
        output = self.post_transformer.fit_transform(output)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)


class DistrSamplePreprocessor(HierarchyPreprocessorBase):
    name = 'DistrSample'

    @staticmethod
    def __take_sample(column):
        uniques,counts = np.unique(column,return_counts=True)
        probs = counts/len(column)
        rng = np.random.default_rng()
        return rng.choice(uniques, len(column),p=probs)

    def __init__(self, generalization):
        self.post_transformer = MinMaxScaler(feature_range=(0, 1))
        super().__init__(generalization, 'DistrSample')

    def train_transform(self, X, y=None):
        X['generalization'] = self.generalization
        output = X.groupby(by='generalization').transform(self.__take_sample)
        output = self.post_transformer.fit_transform(output)
        return output

    def test_transform(self, X, y=None):
        return self.post_transformer.transform(X)
