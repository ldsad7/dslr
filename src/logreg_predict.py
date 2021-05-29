import csv
import json

import click
import numpy as np
import pandas as pd

from log_reg import LogReg
from utils import read_dataset


@click.command()
@click.argument("path_to_dataset")
@click.option("--target_column", '-tc', default="Hogwarts House", help="target feature name which we predict")
@click.option("--exclude_columns", '-ec', default="Astronomy", help="features to exclude from the given dataset")
@click.option("--file_with_params", '-fwpa', default="params.json", help="the file to which we saved learnt weights")
@click.option("--separator", '-s', default=",", help="separator in the file")
@click.option("--verbose", '-v', is_flag=True, default=False, help="verbose output")
@click.option("--file_with_predictions", '-fwpr', default="houses.csv", help="the file to which we save predictions")
def predict(path_to_dataset: str, target_column: str = 'Hogwarts House', exclude_columns: str = "Astronomy",
            file_with_params='params.json', separator: str = ",", verbose: bool = False,
            file_with_predictions: str = "houses.csv") -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    features = df.select_dtypes(include=[np.number]).drop(
        ['Index', target_column, *map(str.strip, exclude_columns.split(','))],
        axis=1, errors='ignore').fillna(df.mean(axis=0))
    features = np.hstack((np.ones((features.shape[0], 1)), features))

    if verbose:
        print(f"reading params from the file `{file_with_params}`")
    with open(file_with_params, 'r', encoding='utf-8') as f:
        data = json.load(f)
        features_mean = np.array(data['features_mean'])
        features_std = np.array(data['features_std'])
        coefficients = np.array(data['coefficients'])
        target_values = np.array(data['target_values'])

    error_message = f"Inconsistent or incorrect data: {data}"
    assert coefficients.shape[0] == target_values.shape[0] and target_values.shape[0] > 0, error_message
    assert features.shape[1] == features_mean.shape[0] == features_std.shape[0] == coefficients[0].shape[0], \
        error_message

    features = (features - features_mean) / (features_std + 10e-10)

    log_reg = LogReg(verbose=verbose)

    results = np.empty(0)
    for i, target_value in enumerate(target_values):
        result = log_reg.predict(features, coefficients[i])
        if not results.shape[0]:
            results = result
        else:
            results = np.hstack((results, result))
    results = [target_values[result] for result in np.argmax(results, axis=1)]
    with open(file_with_predictions, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Index', target_column])
        writer.writerows(list(enumerate(results)))


if __name__ == '__main__':
    try:
        predict()
    except Exception as e:
        print(f'Error happened: {e}')
