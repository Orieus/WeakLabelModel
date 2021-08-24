.PHONY: venv

pip:
	pip install --upgrade pip

venv:
	python3.8 -m venv venv

requirements: pip
	pip install -r requirements38.txt

requirements-dev: requirements pip
	pip install -r requirements-dev.txt

build: requirements-dev
	python3.8 setup.py sdist

pypi: build check-readme
	twine upload dist/*

doc: requirements-dev
	cd docs; make clean; make html

# From Scikit-learn
code-analysis:
	flake8 weaklabels | grep -v external
	pylint -E weaklabels/ -d E1103,E0611,E1101

clean:
	rm -rf ./dist

# All the following assume the requirmenets-dev are installed, but to make the
# output clean the dependency has been removed
test:
	pytest --doctest-modules --cov-report=term-missing --cov=weaklabels weaklabels

check-readme:
	twine check dist/*

