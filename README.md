# ggpy

![Static Badge](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)

A Gaussian Process package for Python, built on [JAX](https://docs.jax.dev).

Kernels and models are immutable pytrees, so they pass straight through `jit`, `grad`
and `vmap`. Fitting returns a new model rather than mutating one, and sampling takes
an explicit PRNG key.

-----

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Testing](#testing)
- [License](#license)

## Features
- Heteroscedastic kernels
- Augmented kernels
- Independent multi-output GPs (`MultiGP` over a `DiagKernel` such as `DiagRBF`),
  which factorizes one matrix per output dimension instead of one big joint matrix
- Parameter optimization by gradient descent (autodiff through the marginal likelihood, via optax)
- Noisy-input regression [WIP]

Note that float64 is enabled process-wide on import (`jax_enable_x64`); GP covariance
factorization needs the precision.

## Usage

```python
import jax
import jax.numpy as jnp
from ggpy import GP, RBF

k_sample = jax.random.key(0)

kernel = RBF(input_dims=1, noise=1e-2, length_scale=2.0, scale=1.0)
model = GP(kernel)

x = jnp.linspace(0, 10, 100)[:, None]
y = model.sample(k_sample, x)          # sampling takes an explicit key

model = model.add_points(x, y, fit=True)   # returns a NEW model
mean, cov = model.predict(x, var=True)

model = model.optimize(steps=200)      # fit hyperparameters, returns a new model
print(model.kernel.length_scale)
```

## Installation

```console
pip install git+https://github.com/Beau-Coup/ggpy
```

## Testing
Install the test extra and run
```console
pip install -e '.[test,gui]'
pytest
```

## License

`ggpy` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
