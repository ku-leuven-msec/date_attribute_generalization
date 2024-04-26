import numpy as np

def initialize(input_df, columns, multipliers=None):
    data = input_df[columns].copy()
    if multipliers is None:
        multipliers = np.ones(len(columns))

    for column, multiplier in zip(columns, multipliers):
        col = data[column]
        data[f'{column} ml'] = col * multiplier

    return data
