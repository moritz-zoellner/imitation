"""cs592_proj: implementations of imitation and reward learning algorithms."""

from importlib import metadata

try:
    __version__ = metadata.version("cs592_proj")
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
