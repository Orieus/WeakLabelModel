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
Usage: testWLCkeras.py [options]

Options:
  -h, --help            show this help message and exit
  -p PROBLEMS, --problems=PROBLEMS
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
  -e METHOD, --method=METHOD
                        Method to generate the matrix M.One of the following:
                        IPL, quasi_IPL, noisy, random_noise, random_weak
  -t METHOD2, --method2=METHOD2
                        Method to impute the matrix M.One of the following:
                        Mproper
```
