"""Project PHANTOM: privacy-preserving affect-aware interaction research prototype."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("project-phantom")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.2.0"

__all__ = ["__version__"]
