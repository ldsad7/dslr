from functools import partial
from math import isnan
from typing import List, Tuple, Callable

import click
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from utils import read_dataset

if __name__ == '__main__':
    from describe_funcs import (
        count, mean, std, min_, first_percentile, second_percentile, third_percentile, max_, amplitude
    )
else:
    from .describe_funcs import (
        count, mean, std, min_, first_percentile, second_percentile, third_percentile, max_, amplitude
    )


def get_column_names_and_values(
        df: pd.DataFrame, rows: List[str], functions: List[Callable]) -> Tuple[List[str], List[List[float]]]:
    values: List[List[float]] = [[] for _ in range(len(rows))]
    columns: List[str] = []
    for column in df:
        if is_numeric_dtype(df[column]):
            for i, (row, function) in enumerate(zip(rows, functions)):
                values[i].append(function(df[column]))
            columns.append(column)
    return columns, values


def print_output(columns: List[str], values: List[List[float]], rows: List[str]) -> None:
    max_lengths = [max(len(row) for row in rows)]
    before_dot_max_lengths = []
    after_dot_max_lengths = []
    for i, column in enumerate(columns):
        before_dot_lengths, after_dot_lengths = [[] for _ in range(2)]
        for lst in values:
            num_as_str: str = "%.6f" % lst[i]
            if '.' in num_as_str:
                before_dot, after_dot = num_as_str.split('.', 1)
            else:
                before_dot, after_dot = num_as_str, ''
            if before_dot.lstrip("-") != 'nan':
                before_dot_lengths.append(len(before_dot.lstrip("-")))
            after_dot_lengths.append(len(after_dot.rstrip('0')) or 1)
        before_dot_max_lengths.append(max(before_dot_lengths))
        after_dot_max_lengths.append(max(after_dot_lengths))
        max_lengths.append(max(len(column), max(before_dot_lengths) + max(after_dot_lengths) + 1))
    print(f"%{max_lengths[0]}s" % "", end="")
    for i, column in enumerate(columns):
        print(f"%{max_lengths[i + 1] + 2}s" % column, end="")
    print()
    for i, row in enumerate(rows):
        print(f"%-{max_lengths[0]}s" % row, end="")
        for j, column in enumerate(columns):
            if isnan(values[i][j]):
                print(f"%{max_lengths[j + 1] + 2}s" % 'NaN', end="")
            else:
                print(f"%{max_lengths[j + 1] + 2}.{after_dot_max_lengths[j]}f" % values[i][j], end="")
        print()


def count_and_print_describe_output(df: pd.DataFrame, add_fields: bool = False) -> None:
    rows: List[str] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    functions: List[Callable] = [
        partial(count), partial(mean), partial(std), partial(min_), partial(first_percentile),
        partial(second_percentile), partial(third_percentile), partial(max_)
    ]

    if add_fields:
        rows.extend(['ucount', 'umean', 'ustd', 'ampl'])
        functions.extend([
            partial(count, unique=True), partial(mean, unique=True), partial(std, unique=True), partial(amplitude)
        ])

    columns, values = get_column_names_and_values(df, rows, functions)
    if not columns:
        raise ValueError("Given dataset doesn't contain data")  # though pandas returns some data frame in this case
    print_output(columns, values, rows)


@click.command()
@click.argument("path_to_dataset")
@click.option("--separator", '-s', default=",", help="separator in the file")
@click.option("--verbose", '-v', is_flag=True, default=False, help="verbose output")
@click.option("--add_fields", '-a', is_flag=True, default=False, help="add additional fields to the describe output")
def describe(path_to_dataset: str, separator: str = ",", verbose: bool = False, add_fields: bool = False) -> None:
    """`describe` reproduces pandas's describe method"""

    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)
    count_and_print_describe_output(df, add_fields)


if __name__ == '__main__':
    try:
        describe()
    except Exception as e:
        print(f'Error happened: {e}')
