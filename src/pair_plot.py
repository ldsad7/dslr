from math import sqrt, ceil
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe_funcs import mean, std

from seaborn import pairplot

COLORS = ['g', 'r', 'y', 'b', 'c']


@click.command()
@click.argument("path_to_dataset")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
@click.option("--save_image", is_flag=True, default=False, help="save image")
def draw_histogram(path_to_dataset: str, separator: str = ",", verbose: bool = False, save_image: bool = False) -> None:
    if verbose:
        print(f'File on the path "{path_to_dataset}" expect the separator `{separator}`')
    df: Optional[pd.DataFrame] = None
    try:
        df = pd.read_csv(path_to_dataset, sep=separator)
    except pd.errors.ParserError as ex:
        print(f'File on the path "{path_to_dataset}" is incorrect ({ex})')
        exit(1)

    unique_houses = df['Hogwarts House'].unique()
    columns = df.select_dtypes(include=[np.number]).drop("Index", axis=1).columns.tolist()

    size = ceil(sqrt(len(columns)))
    fig, axs = plt.subplots(size, size, figsize=(14, 10), dpi=90)  # , sharex='col', sharey='row'
    index = 0

    pairplot(df[columns])

    # for column in columns:
    #     values = df[column]
    #     normalized_values = (values - mean(values)) / std(values)
    #     ax_index = axs[index // size, index % size]
    #     for i, house in enumerate(unique_houses):
    #         color = COLORS[i % len(COLORS)]
    #         house_values = normalized_values[df['Hogwarts House'] == house]
    #         ax_index.hist(
    #             house_values, 50, density=True, facecolor=color, alpha=0.35, edgecolor='black', linewidth=1.2,
    #             label=house
    #         )
    #     ax_index.set_title(column)
    #     index += 1
    # fig.suptitle("Score distribution between Hogwarts houses in different courses", fontsize="x-large")
    #
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower right')
    #
    # fig.supxlabel('Normalized scores')
    # fig.supylabel('Frequency')
    #
    # fig.tight_layout()
    #
    # if save_image:
    #     plt.savefig('pictures/histogram.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    try:
        draw_histogram()
    except Exception as e:
        print(f'Произошла ошибка: {e}')
