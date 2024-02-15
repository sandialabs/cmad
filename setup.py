import os
import numpy as np
from setuptools import setup, find_packages


def read(filename):
    with open(filename, "r") as f:
        long_description = f.read()


setup(
    name="cmad",
    version="1.0.0",
    author="Tom Seidl",
    author_email="dtseidl@sandia.gov",
    description="Constitutive Models via Automatic Differentiation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/cmad",
    packages=find_packages(),
    python_requires='>=3.9',
    classifiers=["Programming Language :: Python :: 3"],
    include_dirs=[np.get_include()],
    setup_requires=["numpy >= 1.20", "scipy >= 1.1.0"],
    install_requires=[
        "python>=3.8",
        "numpy>=1.20",
        "matplotlib",
        "scipy>=1.1.0",
        "jupyter",
        "pytest",
        "coverage>=6.4",
        "pytest-cov",
        "pip",
        "setuptools",
        "scikit-learn",
        "sympy",
        "jax[cpu]",
    ],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx_automodapi", "sphinx_rtd_theme",
                 "sphinx-gallery", "jupyter"]
    }
)
