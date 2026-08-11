import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from timee import TimeeClassifier, TimeeMultivariateClassifier
from timee.model.model import TimeeModel
from timee.model.mv_adapter import TimeeMultivariateModel
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


def _xy_multivariate(n_classes, n_channels=3):
    X_train = RNG.standard_normal((N_TRAIN, n_channels, T)).astype(np.float32)
    y_train = np.array([i % n_classes for i in range(N_TRAIN)])
    X_test = RNG.standard_normal((N_TEST, n_channels, T)).astype(np.float32)
    return X_train, y_train, X_test


def test_multivariate_channel_averaging(clf):
    # TimeeClassifier handles n_channels > 1 by averaging per-channel predictions.
    preds, probs = clf.predict(*_xy_multivariate(4))
    assert preds.shape == (N_TEST,)
    assert probs.shape == (N_TEST, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_multivariate_pooling():
    clf_mv = TimeeMultivariateClassifier(
        model=TimeeMultivariateModel(),
        device=torch.device("cpu"),
        transforms=None,
    )
    preds, probs = clf_mv.predict(*_xy_multivariate(4))
    assert preds.shape == (N_TEST,)
    assert probs.shape == (N_TEST, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_pooling_uv_checkpoint_errors(tmp_path):
    # A univariate checkpoint has no trained variate_attn_pool weights -> hard error.
    save_file(TimeeModel().state_dict(), str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="variate_attn_pool"):
        TimeeMultivariateClassifier.from_pretrained(tmp_path, device="cpu")


def test_pooling_requires_mv_model():
    with pytest.raises(TypeError):
        TimeeMultivariateClassifier(model=TimeeModel(), device=torch.device("cpu"))


@pytest.mark.network
def test_released_mv_checkpoint_loads_strict():
    # Guards against architecture/key drift between mv_adapter.py and the released
    # checkpoint: the checkpoint must load with no missing or unexpected keys.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = hf_hub_download("liamsbhoo/timee-multivariate", filename="model.safetensors")
    missing, unexpected = TimeeMultivariateModel().load_state_dict(
        load_file(weights), strict=False
    )
    assert missing == [], f"checkpoint missing keys: {missing}"
    assert unexpected == [], f"checkpoint has unexpected keys: {unexpected}"

    clf = TimeeMultivariateClassifier.from_pretrained(
        "liamsbhoo/timee-multivariate", device="cpu", use_ensemble=False
    )
    preds, probs = clf.predict(*_xy_multivariate(4))
    assert preds.shape == (N_TEST,)
    assert probs.shape == (N_TEST, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
