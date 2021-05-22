import json

import click
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from log_reg import LogReg
from utils import read_dataset


@click.command()
@click.argument("path_to_dataset")
@click.option("--target_column", default="Hogwarts House", help="target feature name in the given dataset")
@click.option("--file_with_params", default="params.json", help="the file to which we save learnt weights")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
@click.option("--learning_rate", default=0.5, help="learning rate")
@click.option("--epsilon", default=10e-5, help="step size epsilon")
def train(path_to_dataset: str, target_column: str = 'Hogwarts House', file_with_params='params.json',
          separator: str = ",", verbose: bool = False, learning_rate=0.5, epsilon: float = 10e-5) -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    try:
        unique_target_values = df[target_column].unique()
    except KeyError as ex:
        raise ValueError(f"There is no such column in the dataset `{path_to_dataset}` as `{target_column}` ({ex})")

    features = df.select_dtypes(include=[np.number]).drop(
        [target_column, 'Index'], axis=1, errors='ignore').fillna(df.mean(axis=0))
    for column in features.columns:
        assert is_numeric_dtype(features[column])
    features = np.hstack((np.ones((features.shape[0], 1)), features))
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features = (features - features_mean) / (features_std + 10e-10)
    target = df[[target_column]]
    assert features.shape[0] == target.shape[0], \
        "Given dataframes with features and target values differ in length"

    log_reg = LogReg(verbose=verbose, learning_rate=learning_rate, epsilon=epsilon)
    coefficients = []
    for target_value in unique_target_values:
        transformed_target = target[target_column].map({target_value: 1}).fillna(0).to_numpy()
        transformed_target = transformed_target.reshape((transformed_target.shape[0], 1))
        coefficients.append(log_reg.fit(features, transformed_target).tolist())

    if verbose:
        print(f"saving params to the file `{file_with_params}`")
    with open(file_with_params, 'w', encoding='utf-8') as f:
        json.dump({
            'features_mean': features_mean.tolist(),
            'features_std': features_std.tolist(),
            'coefficients': coefficients
        }, f)


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f'Произошла ошибка: {e}')
