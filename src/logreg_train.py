import json
from typing import Optional

import click
import numpy as np
import pandas as pd

from log_reg import LogReg
from utils import read_dataset


@click.command()
@click.argument("path_to_dataset")
@click.option("--target_column", '-tc', default="Hogwarts House", help="target feature name in the given dataset")
@click.option("--exclude_columns", '-ec', default="Astronomy", help="features to exclude from the given dataset")
@click.option("--file_with_params", '-fwp', default="params.json", help="the file to which we save learnt weights")
@click.option("--separator", "-s", default=",", help="separator in the file")
@click.option("--verbose", "-v", is_flag=True, default=False, help="verbose output")
@click.option("--learning_rate", "-lr", default=0.5, help="learning rate")
@click.option("--epsilon", "-e", default=10e-5, help="step size epsilon")
@click.option("--batch_size", "-b", type=click.FloatRange(0, 1), default=1,
              help="batch size in percents from 0 to 1; 0 is equal to sgd, 1 is equal to gd, "
                   "(0, 1) is equal to mini-batch gd")
@click.option("--validation_size", "-vs", type=click.FloatRange(0, 1), default=0.1,
              help="validation set size in percents from 0 to 1 of training dataset")
@click.option("--max_num_of_losses", "-mnol", type=click.IntRange(1), default=20,
              help="maximum number of losses; always positive")
@click.option("--save_to_image", '-sti', default=None, help="save to this image")
def train(path_to_dataset: str, target_column: str = 'Hogwarts House', exclude_columns: str = "Astronomy",
          file_with_params: str = 'params.json', separator: str = ',', verbose: bool = False,
          learning_rate: float = 0.5, epsilon: float = 10e-5, batch_size: float = 1.0, validation_size: float = 0.1,
          max_num_of_losses: int = 20, save_to_image: Optional[str] = None) -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    try:
        unique_target_values = df[target_column].unique()
    except KeyError as ex:
        raise ValueError(f"There is no such column in the dataset `{path_to_dataset}` as `{target_column}` ({ex})")

    features = df.select_dtypes(include=[np.number]).drop(
        [target_column, 'Index', *map(str.strip, exclude_columns.split(','))],
        axis=1, errors='ignore').fillna(df.mean(axis=0))
    features = np.hstack((np.ones((features.shape[0], 1)), features))
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features = (features - features_mean) / (features_std + 10e-10)
    target = df[[target_column]]
    assert features.shape[0] == target.shape[0], \
        "Given dataframes with features and target values differ in length"

    log_reg = LogReg(verbose=verbose, learning_rate=learning_rate, epsilon=epsilon, batch_size=batch_size,
                     validation_size=validation_size, max_num_of_losses=max_num_of_losses)
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
            'coefficients': coefficients,
            'target_values': unique_target_values.tolist()
        }, f)
    if save_to_image is not None:
        log_reg.draw_graph(unique_target_values, save_to_image=save_to_image)


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f'Error happened: {e}')
