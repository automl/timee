"""Preprocessing transforms for the TIMEE inference ensemble.

Each transform has signature: (X_train, X_test) -> (X_train', X_test')
where X arrays are (n_samples, n_channels, seq_len) float32 numpy arrays.
"""

import numpy as np
from scipy.interpolate import interp1d


def first_difference(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace each time series with its first-order difference (x[t] - x[t-1])."""
    return np.diff(X_train, axis=-1), np.diff(X_test, axis=-1)


def interpolate(target_len: int):
    """Return a transform that resamples all series to exactly target_len."""

    def transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        def _resample(X: np.ndarray) -> np.ndarray:
            t_old = np.linspace(0, 1, X.shape[-1])
            t_new = np.linspace(0, 1, target_len)
            return interp1d(t_old, X, kind="linear", axis=-1)(t_new).astype(X.dtype)

        return _resample(X_train), _resample(X_test)

    return transform


def compose_transforms(*transforms):
    """Compose multiple transforms left-to-right."""

    def transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for t in transforms:
            X_train, X_test = t(X_train, X_test)
        return X_train, X_test

    return transform


def default_ensemble_transforms() -> list:
    """Return the 4 transforms used in TIMEE's inference ensemble.

    Members:
      1. interpolate to 256
      2. interpolate to 512
      3. interpolate to 256, then first_difference
      4. interpolate to 512, then first_difference
    """
    return [
        interpolate(target_len=256),
        interpolate(target_len=512),
        compose_transforms(interpolate(target_len=256), first_difference),
        compose_transforms(interpolate(target_len=512), first_difference),
    ]
