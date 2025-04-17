# Timestep prediction using machine learning

This repository is designed to estimate the number of times a timestep length
should be halved in order for a given solver to attain convergence. For solvers
with adaptive timestepping functionality, this allows the user to predict the
timestep length that can be used at each step in the timestepping scheme. If
this prediction is accurate then it avoids the need for trial-and-error
approaches, whereby successively halved timestep lengths are tried until the
solver converges.

## Installation

We strongly advise that users of `mlstep` create a Python virtual environment
before installing it. Doing so avoids polluting the system Python environment.
See https://docs.python.org/3/library/venv.html for details on how to do this.

For a basic install the `mlstep` module, activate your virtual environment,
clone the repository, and then run
```sh
cd mlstep
pip install -e .
```

For a development install, some further steps are recommended:
```sh
cd mlstep

# Install optional dev dependencies
pip install -e .[dev]

# Configure pre-commit hooks
pre-commit install
```
