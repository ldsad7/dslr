import click
import pandas as pd

from utils import read_dataset


@click.command()
@click.option("--our_answers", '-o', default="houses.csv", help="the file with our answers to the task")
@click.option("--true_answers", '-t', default="datasets/dataset_truth.csv",
              help="the file with true answers to the task")
@click.option("--target_column", '-tc', default="Hogwarts House", help="target feature name which we predict")
@click.option("--separator", '-s', default=",", help="separator in the file")
@click.option("--verbose", '-v', is_flag=True, default=False, help="verbose output")
def evaluate(our_answers: str, true_answers: str, target_column: str = 'Hogwarts House',
             separator: str = ",", verbose: bool = False) -> None:
    our_df: pd.DataFrame = read_dataset(our_answers, separator, verbose)
    true_df: pd.DataFrame = read_dataset(true_answers, separator, verbose)
    boolean_df: pd.DataFrame = our_df[our_df[target_column] == true_df[target_column]]

    accuracy: float = boolean_df.shape[0] / true_df.shape[0]
    print(f'current accuracy between "{our_answers}" and "{true_answers}": {accuracy}')


if __name__ == '__main__':
    try:
        evaluate()
    except Exception as e:
        print(f'Error happened: {e}')
