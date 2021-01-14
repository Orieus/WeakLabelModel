from setuptools import setup
from distutils.util import convert_path
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('weaklabels/__init__.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
  name = 'weaklabels',
  packages = ['weaklabels'],
  install_requires=[
    'numpy',
    'scikit-learn',
    'scipy',
    'cvxpy',
  ],
  version=main_ns['__version__'],
  description = 'Python library with tools dealing with weak labels.',
  author = 'Jesus Cid Sueiro, Miquel Perello Nieto, Daniel Bacaicoa',
  author_email = 'perello.nieto@gmail.com',
  url = 'https://github.com/Orieus/WeakLabelModel',
  download_url = 'https://github.com/Orieus/WeakLabelModel/archive/{}.tar.gz'.format(main_ns['__version__']),
  keywords = ['machine learning', 'classification', 'supervised', 'weak labels', 'noisy labels'],
  classifiers = [],
  long_description_content_type='text/markdown',
  long_description=long_description
)
