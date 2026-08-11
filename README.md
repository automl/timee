# TIMEE: Time Series Classification via In-Context Learning

[![arXiv](https://img.shields.io/badge/arXiv-2607.07500-b31b1b.svg)](https://arxiv.org/abs/2607.07500)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-liamsbhoo%2Ftimee-yellow)](https://huggingface.co/liamsbhoo/timee)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

TIMEE is a pretrained transformer for time series classification. It classifies test
series in a single forward pass given labeled training examples — no per-dataset
training or fine-tuning required.

## Installation

```bash
pip install timee-ts
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

## Multivariate Support (beta)

> **Note:** TIMEE is trained and evaluated as a univariate classifier — multivariate is not its focus (yet).

As mentioned in the paper, TIMEE supports multivariate input through two mechanisms:
1. **channel-independent**: each channel is classified separately and the per-channel class probabilities are averaged.
2. **late-channel-mixing**: channels are embedded separately before going through an attention pooling layer (acting as a mixer).

**Mechanism (1)** is already implemented in `TimeeClassifier`: pass `(n_samples, n_channels, seq_len)` input (use `n_channels=1` for the univariate case) and it averages over channels automatically.

**Mechanism (2)** is implemented as `TimeeMultivariateClassifier`, which fuses channels via attention pooling and takes the same `(n_samples, n_channels, seq_len)` input.

```python
from timee import TimeeMultivariateClassifier

clf = TimeeMultivariateClassifier.from_pretrained()
predictions, probabilities = clf.predict(X_train, y_train, X_test)
```

## Citation

```bibtex
@misc{küken2026timeeendtoendtimeseries,
      title={TimEE: End-to-end Time Series Classification via In-Context Learning},
      author={Jaris Küken and Shi Bin Hoo and Martin Mráz and Frank Hutter and Lennart Purucker},
      year={2026},
      eprint={2607.07500},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.07500},
}
```
