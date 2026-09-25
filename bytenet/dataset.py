"""PyTorch Dataset for the file-fragment corpus produced by data_prep.py."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from bytenet.byte2image import byte2image


class FragmentDataset(Dataset):
    """Wraps raw byte sectors + labels, applying Byte2Image + augmentation.

    Returns (raw_bytes_float[Ns], image_float[1,H,W], label) triples, i.e.
    exactly what ByteNet's two branches need (Sec. III-B/C).

    The Byte2Image transform (bit-shift + n-gram, Eqs. 2-3) is
    deterministic given the raw bytes, so it is computed once up front
    (`_precompute`) rather than on every `__getitem__` call across every
    epoch -- only the lightweight augmentation (flip / erase / normalize)
    happens per-access.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_gram: int = 4, train: bool = True,
                 random_erase_p: float = 0.5, hflip_p: float = 0.5):
        self.X = X
        self.y = y
        self.n_gram = n_gram
        self.train = train
        self.random_erase_p = random_erase_p
        self.hflip_p = hflip_p
        self._images = self._precompute()

    def _precompute(self) -> np.ndarray:
        imgs = [byte2image(x, n=self.n_gram) for x in self.X]
        return np.stack(imgs).astype(np.float32)  # (N, H, W), values 0-255

    def __len__(self):
        return len(self.X)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Horizontal flip (paper Sec III-B "image augmentation")
        if self.train and np.random.rand() < self.hflip_p:
            img = img[:, ::-1].copy()
        # Random erase: blank a random rectangular patch
        if self.train and np.random.rand() < self.random_erase_p:
            h, w = img.shape
            eh, ew = max(1, h // 6), max(1, w // 3)
            y0 = np.random.randint(0, max(1, h - eh))
            x0 = np.random.randint(0, max(1, w - ew))
            img[y0:y0 + eh, x0:x0 + ew] = np.random.randint(0, 256)
        return img

    def __getitem__(self, idx):
        raw = self.X[idx].astype(np.float32) / 255.0
        img = self._images[idx].copy()  # (H, W), 0-255
        img = self._augment(img)
        img = img / 255.0
        # normalize (image-level mean/std, "image normalization" in paper)
        img = (img - img.mean()) / (img.std() + 1e-6)

        return (
            torch.from_numpy(raw),
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def mixup(raw, img, labels, num_classes, alpha=0.8):
    """Mixup augmentation (paper Eq. 4/Sec III-B): linearly blend a batch
    with a shuffled copy of itself, and soften the labels to match."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(raw.size(0), device=raw.device)
    raw_m = lam * raw + (1 - lam) * raw[perm]
    img_m = lam * img + (1 - lam) * img[perm]
    y_onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    y_m = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return raw_m, img_m, y_m
