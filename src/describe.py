from math import sqrt
from typing import Optional, List

import click
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def print_columns(df: pd.DataFrame) -> None:
    s = "%7s" % ""
    for column in df:
        s
    print(s)


def count(df_column: pd.DataFrame) -> float:
    return sum(1 for _ in df_column)


def mean(df_column: pd.DataFrame) -> float:
    return sum(value for value in df_column) / count(df_column)


def std(df_column: pd.DataFrame) -> float:
    mean_value: float = mean(df_column)
    return sqrt(sum((value - mean_value) ** 2 for value in df_column) / count(df_column))


def min_(df_column: pd.DataFrame) -> float:
    min_value = df_column[0]
    for value in df_column:
        if value < min_value:
            min_value = value
    return min_value


def max_(df_column: pd.DataFrame) -> float:
    max_value = df_column[0]
    for value in df_column:
        if value > max_value:
            max_value = value
    return max_value


def percentile(df_column: pd.DataFrame, percent: float) -> float:
    round(count(df_column) * percent)


def first_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.25)


def second_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.5)


def third_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.75)


@click.command()
@click.argument("path_to_dataset")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
def describe(path_to_dataset: str, separator: str = ",", verbose: bool = False) -> None:
    """`describe` reproduces pandas's describe method"""

    if verbose:
        print(f'File on the path "{path_to_dataset}" expect the separator `{separator}`')
    df: Optional[pd.DataFrame] = None
    try:
        df = pd.read_csv(path_to_dataset, sep=separator)
    except pd.errors.ParserError as e:
        print(f'File on the path "{path_to_dataset}" is incorrect ({e})')
        exit(1)
    # print_columns(df)
    counts: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    first_percentiles: List[float] = []
    second_percentiles: List[float] = []
    third_percentiles: List[float] = []
    maxs: List[float] = []
    for column in df:
        counts.append(count(df[column]))
        means.append(mean(df[column]))
        stds.append(std(df[column]))
        mins.append(min_(df[column]))
        first_percentiles.append(first_percentile(df[column]))
        second_percentiles.append(second_percentile(df[column]))
        third_percentiles.append(third_percentile(df[column]))
        maxs.append(max_(df[column]))
        if is_numeric_dtype(df[column]):
            # print(column, df[column].count())
            pass
    print(df.describe())


if __name__ == '__main__':
    describe()
