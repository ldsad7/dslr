from typing import Optional

import pandas as pd


def read_dataset(path_to_dataset: str, separator: str = ",", verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f'File on the path "{path_to_dataset}" expects the separator `{separator}`')
    df: Optional[pd.DataFrame] = None
    try:
        df = pd.read_csv(path_to_dataset, sep=separator)
    except pd.errors.ParserError as e:
        print(f'File on the path "{path_to_dataset}" is incorrect ({e})')
        exit(1)
    return df
