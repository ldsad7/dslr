from math import ceil
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import pairplot

from utils import read_dataset


@click.command()
@click.argument("path_to_dataset", default="datasets/dataset_train.csv")
@click.option("--step_size", default=7, help="number of columns to draw in one pairplot")
@click.option("--separator", default=",", help="separator in the file")
@click.option("--verbose", is_flag=True, default=False, help="verbose output")
@click.option("--save_to_image", default=None, help="save to this image")
def draw_pair_plot(path_to_dataset: str, step_size: int = 7, separator: str = ",", verbose: bool = False,
                   save_to_image: Optional[str] = None) -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    columns = df.select_dtypes(include=[np.number]).drop("Index", axis=1).columns.tolist()

    num_of_steps = ceil(len(columns) / step_size)
    for i in range(num_of_steps):
        for j in range(i, num_of_steps):
            if verbose:
                print(f"Drawing {i + 1}.{j + 1}-th pair plot...")
            g = pairplot(
                df, hue="Hogwarts House", x_vars=columns[i * step_size:(i + 1) * step_size],
                y_vars=columns[j * step_size:(j + 1) * step_size], diag_kind="hist"
            )
            for ax in g.axes.flatten():
                ax.set_xlabel(ax.get_xlabel(), rotation=5)
                ax.set_ylabel(ax.get_ylabel(), rotation=75)
            g.fig.subplots_adjust(bottom=0.15, left=0.1, top=0.9, right=0.9)
            g.fig.suptitle("Pair plot of corresponding columns", fontsize="x-large")
            g.fig.supxlabel('X Disciplines')
            g.fig.supylabel('Y Disciplines')
            if save_to_image is not None:
                parts = save_to_image.split('.')
                parts[(len(parts) - 2) % len(parts)] += f'{i + 1}_{j + 1}'
                plt.savefig('.'.join(parts), dpi=300)
            plt.show(block=False)
    plt.show()


if __name__ == '__main__':
    try:
        draw_pair_plot()
    except Exception as e:
        print(f'Error happened: {e}')
