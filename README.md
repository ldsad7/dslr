The file `en.subject.pdf` describes the task

Tests:
- run `py -m pytest` command in the root directory

Steps:
- `python -m venv myvenv`
- `source myvenv/bin/activate`
- `python -m pip install -r requirements.txt`

Usages:
- sgd: `py src\logreg_train.py datasets\dataset_train.csv -v -b 0`
- mini-batch gd: `py src\logreg_train.py datasets\dataset_train.csv -v -b 0.2`
- gd: `py src\logreg_train.py datasets\dataset_train.csv -v -b 1`
- predict on test dataset: `py src\logreg_predict.py datasets\dataset_test.csv`
- evaluate results (count accuracy): `py src\evaluate.py`

Docs:
- why to exclude highly correlated features: https://towardsdatascience.com/why-exclude-highly-correlated-features-when-building-regression-model-34d77a90ea8e
- logistic regression explanation: https://en.wikipedia.org/wiki/Logistic_regression#Logistic_model

Pep8:
- `pycodestyle src --ignore=E501`
