import csv
import json

import click
import numpy as np
import pandas as pd

from log_reg import LogReg
from utils import read_dataset


@click.command()
@click.argument("path_to_dataset")
@click.option("--target_column", default="Hogwarts House", help="target feature name which we predict")
@click.option("--file_with_params", default="params.json", help="the file to which we saved learnt weights")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
@click.option("--file_with_predictions", default="houses.csv", help="the file to which we save predictions")
def predict(path_to_dataset: str, target_column: str = 'Hogwarts House', file_with_params='params.json',
            separator: str = ",", verbose: bool = False, file_with_predictions: str = "houses.csv") -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    features = df.select_dtypes(include=[np.number]).drop('Index', axis=1, errors='ignore').fillna(df.mean(axis=0))
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
    assert features.shape == features_mean.shape == features_std.shape == coefficients[0].shape, error_message

    features = (features - features_mean) / (features_std + 10e-10)

    log_reg = LogReg(verbose=verbose)

    results = np.empty(0)
    for i, target_value in enumerate(target_values):
        results = np.vstack(results, log_reg.predict(features, coefficients[i]))
    print(results)
    results = list(enumerate(results))

    with open(file_with_predictions, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Index', target_column])
        writer.writerows(results)


if __name__ == '__main__':
    try:
        predict()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f'Error happened: {e}')
