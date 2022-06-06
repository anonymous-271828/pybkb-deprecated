import os
import sys
import re
import io

from setuptools import find_packages
from setuptools import setup

__version__ = re.search(r'__version__\s*=\s*[\'"]([0-9]*\.[0-9]*\.[0-9]*)[\'"]',
                        io.open('pybkb/_version.py', encoding='utf_8_sig').read()).group(1)

REQUIRED_PACKAGES = [
    'graphviz>=0.8.3',
    'grave>=0.0.3',
    'networkx>=2.4',
    'numpy<1.22,>=1.14.5',
    'matplotlib>=3.0.3',
    'tqdm>=4.43.0',
    'pandas>=0.24.2',
    'pygraphviz>=1.5',
    'mpi4py',
    'compress_pickle',
    'netifaces',
    'lz4',
    'parse',
    'progress',
    'datetime',
    'gurobipy',
    'scipy',
    'scikit-learn',
    'numba',
    'orange3',
    'ipycytoscape',
    'ray',
]

setup(
    name='pybkb',
    version=__version__,
    description='A Python implementation of a Bayesian Knowledge Base and associated reasoning and learning algorithms.',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.8',
    package_data={'pybkb': ['utils/assets/bkb-style.json']},
)

