Constitutive Models via Automatic Differentiation (CMAD)
========

At Sandia, CMAD is SCR# 2985.0

Installation
------------

CMAD is a pure-Python package; all dependencies install from PyPI.

The recommended setup uses uv (https://docs.astral.sh/uv/), which
provisions the Python interpreter and installs CMAD together:

    uv venv --python 3.13
    uv pip install -e ".[dev]"

`-e .` alone is the base install; the `dev` extra adds tests,
linting, and type-checking. Extras are additive and combine with
commas (quote the brackets so the shell does not glob them):

    uv pip install -e ".[dev,cuda12]"

CMAD supports Python 3.10-3.13. A standard venv works too, if you
already have a supported interpreter:

    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

GPU: JAX installs its CPU build by default. For an NVIDIA GPU, add
the CUDA extra matching your toolkit -- `".[cuda12]"` or
`".[cuda13]"` (install one, not both).
