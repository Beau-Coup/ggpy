"""Hyperparameter fitting with optax.

Kernels are pytrees, so ``jax.grad`` of the negative log marginal likelihood returns
a gradient with the same structure and optax can update it directly -- no flattening
or packing of parameter vectors.

Only the fields a kernel names in its ``TRAINABLE`` tuple are fit; everything else is
held fixed, exactly as the old ``hyper_params`` list did.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from . import _config  # noqa: F401  (enables x64)

__all__ = ["neg_log_likelihood", "optimize"]

#: Multiplicative bounds applied to each trainable parameter, relative to its
#: starting value. Mirrors the box constraint the previous L-BFGS-B fit used.
LOWER_FACTOR = 0.01
UPPER_FACTOR = 100.0


def neg_log_likelihood(kernel, x, y):
    """Negative log marginal likelihood of ``(x, y)`` under ``kernel``.

    Dispatches to the kernel's own implementation, so diagonal kernels get the
    cheaper per-output-dimension form.
    """
    return kernel.neg_log_likelihood(x, y)


def _params_of(kernel):
    return {name: getattr(kernel, name) for name in kernel.TRAINABLE}


@partial(jax.jit, static_argnames=("steps",))
def _run(kernel, x, y, params, lower, upper, steps, learning_rate):
    """Projected Adam on the trainable subtree, as a single scanned loop."""
    optimizer = optax.adam(learning_rate)

    def loss(params):
        return neg_log_likelihood(kernel.replace(**params), x, y)

    def step(carry, _):
        params, opt_state = carry
        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Project back into the box the previous L-BFGS-B fit was constrained to.
        params = jax.tree.map(jnp.clip, params, lower, upper)
        return (params, opt_state), value

    (params, _), history = jax.lax.scan(
        step, (params, optimizer.init(params)), None, length=steps
    )
    return params, history


def optimize(kernel, x, y, *, steps: int = 200, learning_rate: float = 1e-2,
             return_history: bool = False):
    """Fit a kernel's trainable hyperparameters by minimizing the NLL.

    Parameters
    ----------
        kernel: the kernel to fit. Left unchanged.
        x: (n x input_dims) training inputs
        y: (n x output_dims) training outputs
        steps: number of Adam steps
        learning_rate: Adam learning rate
        return_history: also return the NLL at every step

    Returns
    -------
        A new kernel with the fitted hyperparameters, or ``(kernel, history)`` when
        ``return_history`` is set.
    """
    if not kernel.TRAINABLE:
        raise TypeError(f"{type(kernel).__name__} declares no trainable parameters")

    params = _params_of(kernel)
    lower = jax.tree.map(lambda p: jnp.minimum(p * LOWER_FACTOR, p * UPPER_FACTOR), params)
    upper = jax.tree.map(lambda p: jnp.maximum(p * LOWER_FACTOR, p * UPPER_FACTOR), params)

    params, history = _run(
        kernel, x, y, params, lower, upper, steps, learning_rate
    )

    fitted = kernel.replace(**params)
    return (fitted, history) if return_history else fitted
