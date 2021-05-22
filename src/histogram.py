from math import sqrt, ceil, isnan
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe_funcs import mean, std
from utils import read_dataset

COLORS = ['g', 'r', 'y', 'b', 'c']


@click.command()
@click.argument("path_to_dataset", default="datasets/dataset_train.csv")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
@click.option("--save_to_image", default=None, help="save to this image")
def draw_histogram(path_to_dataset: str, separator: str = ",", verbose: bool = False,
                   save_to_image: Optional[str] = None) -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    unique_houses = df['Hogwarts House'].unique()
    columns = df.select_dtypes(include=[np.number]).drop("Index", axis=1).columns.tolist()

    size = ceil(sqrt(len(columns)))
    fig, axs = plt.subplots(size, size, figsize=(14, 10), dpi=90)
    index = 0
    for column in columns:
        values = df[column]
        normalized_values = ((values - mean(values)) / std(values)).replace(float('nan'), 0.0)
        ax_index = axs[index // size, index % size]
        for i, house in enumerate(unique_houses):
            color = COLORS[i % len(COLORS)]
            house_values = normalized_values[df['Hogwarts House'] == house]
            ax_index.hist(
                house_values, 50, density=True, facecolor=color, alpha=0.35, edgecolor='black', linewidth=1.2,
                label=house
            )

        ax_index.set_title(column)
        index += 1
    fig.suptitle("Score distribution between Hogwarts houses in different courses", fontsize="x-large")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')

    fig.supxlabel('Normalized scores')
    fig.supylabel('Frequency')

    fig.tight_layout()

    if save_to_image is not None:
        plt.savefig(save_to_image, dpi=300)
    plt.show()


if __name__ == '__main__':
    try:
        draw_histogram()
    except Exception as e:
        print(f'Error happened: {e}')
