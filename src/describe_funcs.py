from math import sqrt, floor, ceil, isnan
from typing import List

import pandas as pd


def count(df_column: pd.DataFrame, unique=False) -> float:
    values = df_column
    if unique:
        values = set(df_column)
    return sum(1 for value in values if not isnan(value))


def mean(df_column: pd.DataFrame, unique=False) -> float:
    if count(df_column) == 0:
        return float('nan')
    values = df_column
    if unique:
        values = set(df_column)
    return sum(value for value in values if not isnan(value)) / count(df_column, unique=unique)


def std(df_column: pd.DataFrame, unique=False) -> float:
    """
    Normalized by N-1 by default (c)
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html#pandas.DataFrame.std
    """

    cnt: float = count(df_column)
    if cnt < 2:
        return float('nan')
    mean_value: float = mean(df_column, unique=unique)
    values = df_column
    if unique:
        values = set(df_column)
    return sqrt(sum((value - mean_value) ** 2 for value in values if not isnan(value)) / (cnt - 1))


def min_(df_column: pd.DataFrame) -> float:
    min_value = float('nan')
    for i in range(df_column.size):
        if not isnan(df_column[i]):
            min_value = df_column[i]
            break
    if isnan(min_value):
        return min_value
    for value in df_column:
        if isnan(value):
            continue
        if value < min_value:
            min_value = value
    return min_value


def max_(df_column: pd.DataFrame) -> float:
    max_value = float('nan')
    for i in range(df_column.size):
        if not isnan(df_column[i]):
            max_value = df_column[i]
            break
    if isnan(max_value):
        return max_value
    for value in df_column:
        if isnan(value):
            continue
        if value > max_value:
            max_value = value
    return max_value


def amplitude(df_column: pd.DataFrame) -> float:
    return max_(df_column) - min_(df_column)


def quick_sort(lst: List[float]) -> List[float]:
    """consider increasing max recursion depth..."""

    less, equal, greater = [[] for _ in range(3)]
    if len(lst) > 1:
        pivot = lst[len(lst) // 2]
        for value in lst:
            if isnan(value):
                continue
            if value < pivot:
                less.append(value)
            elif value > pivot:
                greater.append(value)
            else:
                equal.append(value)
        return quick_sort(less) + equal + quick_sort(greater)
    return lst


def percentile(df_column: pd.DataFrame, percent: float) -> float:
    """
    https://stackoverflow.com/a/2753343/8990391
    """

    assert df_column.size != 0, "df_column should not be empty"
    assert 0.0 <= percent <= 1.0, "percent should be in range [0; 1]"
    lst = [value for value in list(df_column.values) if not isnan(value)]
    if not lst:
        return float('nan')
    sorted_list: List[float] = quick_sort(lst)
    index = (len(sorted_list) - 1) * percent
    if index.is_integer():
        return sorted_list[int(index)]
    lower_index: int = floor(index)
    upper_index: int = ceil(index)
    return sorted_list[lower_index] * (upper_index - index) + sorted_list[upper_index] * (index - lower_index)


def first_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.25)


def second_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.5)


def third_percentile(df_column: pd.DataFrame) -> float:
    return percentile(df_column, 0.75)
