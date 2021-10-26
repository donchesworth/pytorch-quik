from importlib import import_module

__all__ = [
    "api",
    "arg",
    "io",
    "ddp",
    "metrics",
    "serve",
    "tensor",
    "transform",
    "travel",
    "utils",
]
_optional = ["bert", "mlflow", "tune"]
__version__ = "0.3.0"

for submodule in __all__:
    import_module(f"pytorch_quik.{submodule}")

for submodule in _optional:
    try:
        import_module(f"pytorch_quik.{submodule}")
        __all__.append(submodule)
    except ImportError:
        print(f"Error importing {submodule}, skipping functions")
