The file `en.subject.pdf` describes the task

Tests:
- run `py -m pytest` command in the root directory

Steps:
- `python -m venv myvenv`
- `myvenv\Scripts\activate`
- `python -m pip install -r requirements.txt`

Usages:
- describe: `py src\describe.py datasets\dataset_train.csv`
- histogram: `py src\histogram.py datasets\dataset_train.csv`
- pair plot: `py src\pair_plot.py datasets\dataset_train.csv`
- scatter plot: `py src\scatter_plot.py datasets\dataset_train.csv`
- gd: `py src\logreg_train.py datasets\dataset_train.csv -v`
- predict on test dataset: `py src\logreg_predict.py datasets\dataset_test.csv`

Bonuses:
- describe: `py src\describe.py -a datasets\dataset_train.csv`
- sgd: `py src\logreg_train.py datasets\dataset_train.csv -v -b 0`
- mini-batch gd: `py src\logreg_train.py datasets\dataset_train.csv -v -b 0.2 --save_to_image ""`
- evaluate results (count accuracy): `py src\evaluate.py`

Docs:
- why to exclude highly correlated features: https://towardsdatascience.com/why-exclude-highly-correlated-features-when-building-regression-model-34d77a90ea8e
- logistic regression explanation: https://en.wikipedia.org/wiki/Logistic_regression#Logistic_model

Pep8:
- `pycodestyle src --ignore=E501`
