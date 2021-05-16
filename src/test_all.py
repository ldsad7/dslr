import subprocess


class TestParser:
    def test_describe(self):
        for file_path in ["datasets/dataset_train.csv", "datasets/dataset_test.csv",
                          "datasets/dataset_train_1_row.csv", "datasets/dataset_test_1_row.csv"]:
            output_1 = subprocess.check_output(["py", "src/describe.py", file_path])
            output_2 = subprocess.check_output(
                ["py", "-c", f"import pandas as pd;pd.set_option('display.width', 1000);"
                             f"pd.set_option('display.max_columns', 500);"
                             f"df = pd.read_csv('{file_path}');print(df.describe())"])
            assert output_1 == output_2
