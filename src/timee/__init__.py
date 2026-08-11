"""TIMEE: End-to-end time series classification via in-context learning."""

from importlib.metadata import PackageNotFoundError, version

from timee.classifier import TimeeClassifier, TimeeMultivariateClassifier

try:
    __version__ = version("timee-ts")
except PackageNotFoundError:  # not installed, e.g. running from a source checkout
    __version__ = "0.0.0.dev0"

__all__ = ["TimeeClassifier", "TimeeMultivariateClassifier", "__version__"]
