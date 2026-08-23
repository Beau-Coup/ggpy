"""Pytree plumbing shared by every kernel and model in :mod:`ggpy`.

Kernels and models are immutable frozen dataclasses registered as JAX pytrees, so
they can be passed straight through ``jit``, ``grad`` and ``vmap``. Each field is
either

* a **leaf** -- a traced array (hyperparameters, training data, factorizations), or
* **static** -- shape/structure metadata carried in the treedef. Static fields must
  be hashable and are compared by equality when JAX looks up a compiled kernel.

Mark the latter with :func:`static_field`::

    @pytree_dataclass
    class RBF:
        input_dims: int = static_field()
        length_scale: jax.Array = 1.0

Because a class is part of its own treedef, ``jit`` specializes per kernel type
automatically -- no dispatch table required.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from jax.tree_util import register_pytree_node

__all__ = ["pytree_dataclass", "static_field", "set_frozen"]


def static_field(**kwargs):
    """A dataclass field stored in the treedef rather than as a traced leaf."""
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def set_frozen(obj, **values):
    """Assign attributes on a frozen instance.

    Only for use inside a hand-written ``__init__``; everything else should build a
    new instance instead of mutating one.
    """
    for name, value in values.items():
        object.__setattr__(obj, name, value)


def pytree_dataclass(cls=None, *, init=True):
    """Turn ``cls`` into a frozen dataclass registered as a pytree node.

    Parameters
    ----------
    init:
        Pass ``False`` to keep a hand-written ``__init__`` (needed for variadic
        constructors, or when a field is derived from the others). Use
        :func:`set_frozen` to assign fields in such an ``__init__``.
    """

    def wrap(cls):
        cls = dataclass(frozen=True, init=init, eq=False, repr=False)(cls)

        leaf_names = tuple(
            f.name for f in fields(cls) if not f.metadata.get("static", False)
        )
        static_names = tuple(
            f.name for f in fields(cls) if f.metadata.get("static", False)
        )

        def tree_flatten(obj):
            children = tuple(getattr(obj, n) for n in leaf_names)
            aux = tuple(getattr(obj, n) for n in static_names)
            return children, aux

        def tree_unflatten(aux, children):
            # Bypass __init__: during tracing, transposition and `tree_map`
            # placeholders, children may be tracers or sentinel objects that any
            # real constructor logic would choke on.
            obj = object.__new__(cls)
            for name, value in zip(leaf_names, children):
                object.__setattr__(obj, name, value)
            for name, value in zip(static_names, aux):
                object.__setattr__(obj, name, value)
            return obj

        def replace(self, **changes):
            """Return a copy with ``changes`` applied, leaving ``self`` untouched."""
            current = {f.name: getattr(self, f.name) for f in fields(self)}
            current.update(changes)
            return tree_unflatten(
                tuple(current[n] for n in static_names),
                tuple(current[n] for n in leaf_names),
            )

        def __repr__(self):
            body = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
            return f"{type(self).__name__}({body})"

        cls._leaf_names = leaf_names
        cls._static_names = static_names
        cls.tree_flatten = tree_flatten
        cls.tree_unflatten = staticmethod(tree_unflatten)
        cls.replace = replace
        cls.__repr__ = __repr__

        register_pytree_node(cls, tree_flatten, tree_unflatten)
        return cls

    return wrap if cls is None else wrap(cls)
