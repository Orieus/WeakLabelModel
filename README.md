[![CI][ci:b]][ci]
[![License BSD3][license:b]][license]
![Python3.8][python:b]
[![pypi][pypi:b]][pypi]
[![codecov][codecov:b]][codecov]

[ci]: https://github.com/Orieus/WeakLabelModel/actions/workflows/ci.yml
[ci:b]: https://github.com/Orieus/WeakLabelModel/workflows/CI/badge.svg
[license]: https://github.com/Orieus/WeakLabelModel/blob/master/LICENSE.txt
[license:b]: https://img.shields.io/github/license/Orieus/WeakLabelModel.svg
[python:b]: https://img.shields.io/badge/python-3.8-blue
[pypi]: https://badge.fury.io/py/weaklabels
[pypi:b]: https://badge.fury.io/py/weaklabels.svg
[codecov]: https://codecov.io/gh/Orieus/WeakLabelModel
[codecov:b]: https://codecov.io/gh/Orieus/WeakLabelModel/branch/master/graph/badge.svg?token=AYMZPLELT3

# WeakLabelModel
A library for training multiclass classifiers with weak labels

### Installation

```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run code

See example of a call inside one of the add_queue scripts

### Unittest

Run the unit tests running the following script from a terminal

```bash
./runtests.sh
```

### Run Jupyter Notebooks

In order to run any notebook in this repository, first create a new kernel that
will be available from the Jupyter Notebook.

```
# Load the virtual environment
source venv/bin/activate
# Create a new kernel
ipython kernel install --name "weaklabels" --user
# Open Jupyter
jupyter notebook
```

After opening or creating a notebook, you can select the "weaklabels" kernel in
kernel -> Change kernel -> weaklabels


### Usage

Current usage (may need updating)


```
Usage: main.py [options]

Options:
  -h, --help            show this help message and exit
  -p DATASETS, --datasets=DATASETS
                        List of datasets or toy examples totest separated by
                        with no spaces.
  -s NS, --n-samples=NS
                        Number of samples if toy dataset.
  -f NF, --n-features=NF
                        Number of features if toy dataset.
  -c N_CLASSES, --n-classes=N_CLASSES
                        Number of classes if toy dataset.
  -m N_SIM, --n-simulations=N_SIM
                        Number of times to run every model.
  -l LOSS, --loss=LOSS  Loss function to minimize between square (brier score)
                        or CE (cross entropy)
  -u PATH_RESULTS, --path-results=PATH_RESULTS
                        Path to save the results
  -r RHO, --rho=RHO     Learning step for the Gradient Descent
  -a ALPHA, --alpha=ALPHA
                        Alpha probability parameter
  -b BETA, --beta=BETA  Beta probability parameter
  -g GAMMA, --gamma=GAMMA
                        Gamma probability parameter
  -i N_IT, --n-iterations=N_IT
                        Number of iterations of Gradient Descent.
  -e MIXING_MATRIX, --mixing-matrix=MIXING_MATRIX
                        Method to generate the mixing matrix M.One of the
                        following: IPL, quasi-IPL, noisy, random_noise,
                        random_weak

```


### Check that all datasets work

The python code in utils/data.py can be run in order to check that all datasets
can be downloaded, preprocessed and a classifier can be trained on them. This
can be done by calling the python code as a main file

```
python utils/data.py
```

Should output the following for each dataset


```
Testing all datasets
Evaluating iris[61] dataset
----------------
Dataset description:
    Dataset name: iris
    Sample size: 150
    Number of features: 4
    Number of classes: 3
Logistic Regression score = 0.9733333333333334
```

Upload to PyPi
--------------

Create the files to distribute

```
python3.8 setup.py sdist
```

Ensure twine is installed

```
pip install twine
```

Upload the distribution files

```
twine upload dist/*
```

It may require user and password if these are not set in your home directory a
file  __.pypirc__

```
[pypi]
username = __token__
password = pypi-yourtoken
```

Check the README file for Pypi
------------------------------

In order to check that the readme file is compliant with Pypi standards,
install the following Python package

```
pip install readme-renderer
```

and run the following command

```
twine check dist/*
```

Contributors
------------

- Jesus Cid Sueiro
- Miquel Perello Nieto
- Daniel Bacaicoa
