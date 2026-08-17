Constitutive Models via Automatic Differentiation (CMAD)
========

At Sandia, CMAD is SCR# 2985.0

Installation
------------

CMAD is a pure-Python package; all required dependencies install from
PyPI.

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

CPU devices: `cmad --cpu-devices N ...` runs JAX with N CPU devices and
shards the FE assembly across them. The count has to be set before JAX
starts, so it is a command line option, not an input file entry. A
script that imports cmad instead of running the `cmad` command sets it
itself with `jax.config.update("jax_num_cpu_devices", N)` right after
`import jax` (or `XLA_FLAGS=--xla_force_host_platform_device_count=N`).

Sparse direct solver: the `solvers` extra adds scikit-sparse, which
supplies a fill reducing ordering for the sparse direct solve. It
reduces fill and can roughly halve the factorization cost. It is
optional; without it the solve uses SuperLU's COLAMD ordering.

Unlike the other extras it builds against a system SuiteSparse, so
install that first:

    brew install suite-sparse              # macOS
    sudo apt install libsuitesparse-dev    # Debian, Ubuntu
    sudo dnf install suitesparse-devel     # Fedora, RHEL

then

    uv pip install -e ".[solvers]"
