"""P1 components; evaluation manifests remain usable without model dependencies."""
from importlib import import_module

__all__ = ["P1Batch", "P1Config", "P1Model"]


def __getattr__(name):
    modules = {"P1Batch": ".types", "P1Config": ".config", "P1Model": ".models.p1"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
