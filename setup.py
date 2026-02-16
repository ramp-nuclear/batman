"""Install directions.

"""
from pathlib import Path

from setuptools import find_packages

from conda_setup import setup

if __name__ == '__main__':
    setup(name='batman',
          packages=find_packages(),
          scripts=[],
          entry_points={},
          requirements_yml='requirements.yml',
          )
