from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import scatterplot

from utils import read_dataset


@click.command()
@click.argument("path_to_dataset", default="datasets/dataset_train.csv")
@click.option("--separator", '-s', default=",", help="separator in the file")
@click.option("--column_1", '-c1', default="Astronomy", help="X column")
@click.option("--column_2", '-c2', default="Defense Against the Dark Arts", help="Y column")
@click.option("--verbose", '-v', is_flag=True, default=False, help="verbose output")
@click.option("--save_to_image", '-sti', default=None, help="save to this image")
def draw_scatter_plot(path_to_dataset: str, separator: str = ",", column_1: str = "Astronomy",
                      column_2: str = "Defense Against the Dark Arts", verbose: bool = False,
                      save_to_image: Optional[str] = None) -> None:
    df: pd.DataFrame = read_dataset(path_to_dataset, separator, verbose)

    columns = df.select_dtypes(include=[np.number]).drop("Index", axis=1).columns.tolist()

    for column in [column_1, column_2]:
        if column not in columns:
            raise ValueError(f"There is no column '{column}'. Column name should be on of the following: {columns}")

    scatterplot(data=df, x=column_1, y=column_2, hue="Hogwarts House")

    plt.title(f"Scatter plot of '{column_1}' to '{column_2}'", fontsize="x-large")

    if save_to_image is not None:
        if not save_to_image.strip():
            save_to_image = f'pictures/scatter_plot_{"_".join(column_1.split())}_to_{"_".join(column_2.split())}.png'
        plt.savefig(save_to_image, dpi=300)
    plt.show()


if __name__ == '__main__':
    try:
        draw_scatter_plot()
    except Exception as e:
        print(f'Error happened: {e}')
