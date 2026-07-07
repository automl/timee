# TIMEE: Time Series Classification via In-Context Learning

TIMEE is a pretrained transformer for time series classification. It classifies test
series in a single forward pass given labeled training examples — no per-dataset
training or fine-tuning required.

**Paper:** *TIMEE: End-to-end Time Series Classification via In-Context Learning*

## Installation

```bash
pip install timee
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0.

## Quickstart

```python
from timee import TimeeClassifier
import numpy as np

clf = TimeeClassifier.from_pretrained()  # downloads from HuggingFace on first use

# X: (n_samples, n_channels, seq_len) float32
X_train = np.random.randn(20, 1, 256).astype(np.float32)
y_train = np.array([0, 1] * 10)
X_test  = np.random.randn(5, 1, 256).astype(np.float32)

predictions, probabilities = clf.predict(X_train, y_train, X_test)
```

Labels can be any type (`int`, `str`, etc.). Datasets with more than 10 classes are
handled automatically via one-vs-rest.

## API

### `TimeeClassifier.from_pretrained(path, device=None, use_ensemble=True)`

Loads a checkpoint from a directory containing `model.safetensors`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `"liamsbhoo/timee"` | Local directory or HuggingFace Hub repo ID. |
| `device` | auto | `"cuda"`, `"cpu"`, or `torch.device`. Defaults to CUDA > MPS > CPU. |
| `use_ensemble` | `True` | 4-member preprocessing ensemble (interpolate×{256,512} × {raw, diff}). Set `False` for faster single-pass inference. |

### `clf.predict(X_train, y_train, X_test)`

Returns `(predictions, probabilities)`:
- `predictions`: `(n_test,)`, same type as `y_train`
- `probabilities`: `(n_test, n_classes)`, rows sum to 1

## Citation

```bibtex
@inproceedings{timee2026,
  title     = {TIMEE: End-to-end Time Series Classification via In-Context Learning},
  year      = {2026},
}
```
