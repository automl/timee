"""TimeeClassifier: end-to-end time series classification via in-context learning."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from timee.model.model import TimeeModel
from timee.model.mv_adapter import TimeeMultivariateModel
from timee.transforms import default_ensemble_transforms

EnsembleTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_weights_path(path: Union[str, Path]) -> Path:
    """Resolve a checkpoint location to a local ``model.safetensors`` path.

    Accepts a local directory containing ``model.safetensors`` or a HuggingFace Hub repo ID.
    """
    local = Path(path)
    if local.is_dir():
        weights_path = local / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {local}")
        return weights_path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(path), filename="model.safetensors"))


class TimeeClassifier:
    """Pretrained TIMEE model for in-context time series classification.

    A single forward pass classifies test series given labeled training examples.
    No per-dataset training or fine-tuning required.

    Usage::

        from timee import TimeeClassifier

        clf = TimeeClassifier.from_pretrained("path/to/checkpoint/")

        # X arrays: (n_samples, n_channels, seq_len) float32 numpy
        predictions, probabilities = clf.predict(X_train, y_train, X_test)

    Datasets with more than 10 classes are handled automatically via one-vs-rest (OvR).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        transforms: list[EnsembleTransform] | None = None,
    ) -> None:
        self.device = device
        self._model = model.to(device)
        self._model.eval()
        self.transforms = transforms
        self.max_classes = getattr(model, "n_max_classes", None) or getattr(
            model, "n_classes", None
        )
        if self.max_classes is None:
            raise ValueError("Model must have an n_max_classes attribute")

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path] = "liamsbhoo/timee",
        device: Union[str, torch.device, None] = None,
        use_ensemble: bool = True,
    ) -> "TimeeClassifier":
        """Load a pretrained TimeeClassifier.

        Args:
            path: Local directory containing ``model.safetensors``, or a
                  HuggingFace Hub repo ID (e.g. ``"liamsbhoo/timee"``).
                  Defaults to the official checkpoint.
            device: Torch device for inference. Defaults to auto-detection (CUDA > MPS > CPU).
            use_ensemble: If True (default), use the 4-member preprocessing ensemble from the
                paper (interpolate×{256,512} × {raw, first_difference}).
                Set to False for faster single-pass inference without preprocessing.

        Returns:
            A ready-to-use TimeeClassifier.
        """
        if device is None:
            device = _infer_device()
        device = torch.device(device) if isinstance(device, str) else device

        weights_path = _resolve_weights_path(path)

        from safetensors.torch import load_file

        model = TimeeModel()
        model.load_state_dict(load_file(weights_path))

        transforms = default_ensemble_transforms() if use_ensemble else None
        return cls(model=model, device=device, transforms=transforms)

    def _prepare_inputs(
        self,
        X_train: np.ndarray,
        y_train_encoded: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_np = np.concatenate([X_train, X_test], axis=0)
        targets_np = np.concatenate(
            [y_train_encoded.astype(np.int64), np.zeros(len(X_test), dtype=np.int64)],
            axis=0,
        )

        if inputs_np.ndim == 3 and inputs_np.shape[1] > 1:
            # Multivariate (N, C, T) -> (N, T, C) -> (1, N, T, C)
            inputs = (
                torch.from_numpy(inputs_np.transpose(0, 2, 1)).float().to(self.device).unsqueeze(0)
            )
        else:
            # Univariate (N, 1, T) -> squeeze -> (N, T) -> (1, N, T)
            inputs = torch.from_numpy(inputs_np).float().to(self.device).squeeze().unsqueeze(0)

        targets = torch.from_numpy(targets_np).long().to(self.device).unsqueeze(0)
        return inputs, targets

    def _forward(
        self,
        X_train: np.ndarray,
        y_enc: np.ndarray,
        X_test_batch: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        inputs, targets = self._prepare_inputs(X_train, y_enc, X_test_batch)
        logits = self._model(x=inputs, y=targets, eval_pos=len(X_train)).squeeze(0)[:, :n_classes]
        return torch.softmax(logits, dim=1).cpu().numpy()

    def _forward_ovr(
        self,
        X_train: np.ndarray,
        y_enc: np.ndarray,
        X_test_batch: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        inputs, targets = self._prepare_inputs(X_train, y_enc, X_test_batch)
        eval_pos = len(X_train)
        class_probs = []
        for c in range(n_classes):
            targets_binary = torch.zeros_like(targets)
            targets_binary[:, :eval_pos] = (targets[:, :eval_pos] == c).long()
            logits = self._model(x=inputs, y=targets_binary, eval_pos=eval_pos).squeeze(0)[:, :2]
            class_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[:, 1])
        probabilities = np.column_stack(class_probs)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def _predict_batched(
        self,
        X_train: np.ndarray,
        y_enc: np.ndarray,
        X_test: np.ndarray,
        n_classes: int,
        batch_fn: Callable,
    ) -> np.ndarray:
        n_test = len(X_test)
        batch_size = n_test
        while batch_size >= 1:
            try:
                chunks = []
                for start in range(0, n_test, batch_size):
                    chunks.append(
                        batch_fn(X_train, y_enc, X_test[start : start + batch_size], n_classes)
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                return np.concatenate(chunks, axis=0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size //= 2
        raise RuntimeError(
            "GPU OOM even with a single test sample -- "
            "training context alone exceeds available memory."
        )

    def _predict_probabilities(
        self,
        X_train: np.ndarray,
        y_enc: np.ndarray,
        X_test: np.ndarray,
        n_classes: int,
        batch_fn: Callable,
    ) -> np.ndarray:
        """Return (n_test, n_classes) probabilities, averaged over the preprocessing ensemble."""
        x_transforms: list[EnsembleTransform | None] = self.transforms or [None]

        with torch.no_grad():
            all_probs = []
            for x_transform in x_transforms:
                X_tr_t, X_te_t = x_transform(X_train, X_test) if x_transform else (X_train, X_test)
                all_probs.append(self._predict_batched(X_tr_t, y_enc, X_te_t, n_classes, batch_fn))

        return np.mean(all_probs, axis=0)

    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify test samples given labeled training examples.

        Multivariate input (``n_channels > 1``) is handled channel-independently: each
        channel is classified separately and the per-channel class probabilities are
        averaged. For a jointly-modeled multivariate approach with attention-based variate
        pooling, use :class:`TimeeMultivariateClassifier`.

        Args:
            X_train: Training time series, shape (n_train, n_channels, seq_len).
            y_train: Training labels, shape (n_train,). Any hashable label type.
            X_test: Test time series, shape (n_test, n_channels, seq_len).

        Returns:
            predictions: Predicted class labels, shape (n_test,). Same dtype as y_train.
            probabilities: Class probability estimates, shape (n_test, n_classes).
        """
        le = LabelEncoder().fit(y_train)
        n_classes = len(le.classes_)
        y_enc = le.transform(y_train)

        batch_fn = self._forward_ovr if n_classes > self.max_classes else self._forward

        n_channels = X_train.shape[1]
        if n_channels == 1:
            probabilities = self._predict_probabilities(
                X_train, y_enc, X_test, n_classes, batch_fn
            )
        else:
            # Channel-independent: classify each channel, average the class probabilities.
            channel_probs = [
                self._predict_probabilities(
                    X_train[:, c : c + 1, :], y_enc, X_test[:, c : c + 1, :], n_classes, batch_fn
                )
                for c in range(n_channels)
            ]
            probabilities = np.mean(channel_probs, axis=0)

        return le.inverse_transform(np.argmax(probabilities, axis=1)), probabilities


class TimeeMultivariateClassifier(TimeeClassifier):
    """Multivariate TIMEE classifier using attention-based variate pooling.

    Wraps :class:`~timee.model.mv_adapter.TimeeMultivariateModel`, which encodes each
    variate through the shared univariate encoder and fuses the per-variate representations
    with learned attention pooling before the in-context phase — modeling the channels
    jointly rather than independently.

    This requires a checkpoint **fine-tuned for multivariate pooling** (i.e. one that
    contains trained ``variate_pool`` weights); the released univariate checkpoint has none.
    For zero-shot multivariate classification without fine-tuning, use
    :class:`TimeeClassifier`, which averages per-channel predictions.

    Usage::

        from timee import TimeeMultivariateClassifier

        clf = TimeeMultivariateClassifier.from_pretrained()
        predictions, probabilities = clf.predict(X_train, y_train, X_test)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        transforms: list[EnsembleTransform] | None = None,
    ) -> None:
        if not isinstance(model, TimeeMultivariateModel):
            raise TypeError(
                "TimeeMultivariateClassifier requires a TimeeMultivariateModel, got "
                f"{type(model).__name__}. Use TimeeClassifier for univariate models."
            )
        super().__init__(model=model, device=device, transforms=transforms)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path] = "liamsbhoo/timee-multivariate",
        device: Union[str, torch.device, None] = None,
        use_ensemble: bool = True,
    ) -> "TimeeMultivariateClassifier":
        """Load a pretrained multivariate (variate-pooling) classifier.

        Args:
            path: Local directory containing ``model.safetensors``, or a HuggingFace Hub
                  repo ID (e.g. ``"liamsbhoo/timee-multivariate"``). Must be a checkpoint
                  fine-tuned for multivariate pooling (i.e. containing ``variate_attn_pool``
                  weights). Defaults to the official multivariate checkpoint.
            device: Torch device for inference. Defaults to auto-detection (CUDA > MPS > CPU).
            use_ensemble: If True (default), use the 4-member preprocessing ensemble.

        Returns:
            A ready-to-use TimeeMultivariateClassifier.

        Raises:
            ValueError: if the checkpoint contains no trained ``variate_pool`` weights.
        """
        if device is None:
            device = _infer_device()
        device = torch.device(device) if isinstance(device, str) else device

        weights_path = _resolve_weights_path(path)

        from safetensors.torch import load_file

        model = TimeeMultivariateModel()
        missing_keys, _ = model.load_state_dict(load_file(weights_path), strict=False)
        if any(k.startswith("variate_attn_pool") for k in missing_keys):
            raise ValueError(
                f"Checkpoint at {path!r} has no trained 'variate_attn_pool' weights, so it "
                "cannot be used for attention-based variate pooling. Provide a checkpoint "
                "fine-tuned for multivariate pooling, or use TimeeClassifier for zero-shot "
                "channel-independent classification."
            )

        transforms = default_ensemble_transforms() if use_ensemble else None
        return cls(model=model, device=device, transforms=transforms)

    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify multivariate test samples via joint variate pooling.

        Args:
            X_train: Training time series, shape (n_train, n_channels, seq_len).
            y_train: Training labels, shape (n_train,). Any hashable label type.
            X_test: Test time series, shape (n_test, n_channels, seq_len).

        Returns:
            predictions: Predicted class labels, shape (n_test,). Same dtype as y_train.
            probabilities: Class probability estimates, shape (n_test, n_classes).
        """
        le = LabelEncoder().fit(y_train)
        n_classes = len(le.classes_)
        y_enc = le.transform(y_train)

        batch_fn = self._forward_ovr if n_classes > self.max_classes else self._forward
        probabilities = self._predict_probabilities(X_train, y_enc, X_test, n_classes, batch_fn)
        return le.inverse_transform(np.argmax(probabilities, axis=1)), probabilities
