import numpy as np
import pytest
import torch

from timee import TimeeClassifier
from timee.model.model import TimeeModel
from timee.transforms import default_ensemble_transforms

RNG = np.random.default_rng(42)
N_TRAIN, N_TEST, C, T = 6, 3, 1, 64


@pytest.fixture(scope="module")
def clf():
    model = TimeeModel()
    return TimeeClassifier(model=model, device=torch.device("cpu"), transforms=None)


def _xy(n_classes):
    X_train = RNG.standard_normal((N_TRAIN, C, T)).astype(np.float32)
    y_train = np.array([i % n_classes for i in range(N_TRAIN)])
    X_test = RNG.standard_normal((N_TEST, C, T)).astype(np.float32)
    return X_train, y_train, X_test


def test_binary(clf):
    preds, probs = clf.predict(*_xy(2))
    assert preds.shape == (N_TEST,)
    assert probs.shape == (N_TEST, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass(clf):
    preds, probs = clf.predict(*_xy(4))
    assert probs.shape == (N_TEST, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_string_labels(clf):
    X_train = RNG.standard_normal((N_TRAIN, C, T)).astype(np.float32)
    y_train = np.array(["cat", "cat", "cat", "dog", "dog", "dog"])
    X_test = RNG.standard_normal((N_TEST, C, T)).astype(np.float32)
    preds, _ = clf.predict(X_train, y_train, X_test)
    assert all(p in {"cat", "dog"} for p in preds)


def test_ensemble():
    model = TimeeModel()
    clf_ens = TimeeClassifier(
        model=model,
        device=torch.device("cpu"),
        transforms=default_ensemble_transforms(),
    )
    preds, probs = clf_ens.predict(*_xy(2))
    assert preds.shape == (N_TEST,)
    assert probs.shape == (N_TEST, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
