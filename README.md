#  `crospint`: Coordinate Rotation for Spatial Interpolation

This package provides lightweight utilities and `scikit-learn`-compatible transformers to build efficient **machine learning pipelines for spatio-temporal interpolation** with tree-based ensemble methods. This package also contains functions and classes uses to model housing prices at the national scale (see Meslin [2026]).

This package has three distinctive features:

- It offers a native `sklearn.Pipeline` integration;
- It uses [`polars`](https://pola.rs/) for fast feature engineering but is also fully [`pandas`](https://pandas.pydata.org/)-compatible to simplify use with other ML libraries;
- It is agnostic to the algorithms used for interpolation (though is uses [`lightgbm`](https://lightgbm.readthedocs.io/en/stable/) by default).

This package also contains functions and classes uses to model housing prices at the national scale (see Meslin [2026]).

## Installation

Just execute `uv add crospint` or `pip install crospint`.

## Basic use

To be completed

# Notebooks

Two notebooks illustrate the valuation method describe in Meslin [2026]:

- How to apply the conditional tail removal procedure [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliviermeslin/crospint/blob/improve_readme/notebooks/outlier_removal_through_CTR.ipynb)

- How to train price models at the national scale [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliviermeslin/crospint/blob/improve_readme/notebooks/model_training_and_evaluation.ipynb)


