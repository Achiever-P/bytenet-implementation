"""
Byte2Image representation model.

Implements the visual representation described in:
  Liu et al., "ByteNet: Rethinking Multimedia File Fragment Classification
  Through Visual Perspectives," IEEE TMM, 2025.

Pipeline (paper Section III-B, Eqs. 2-4, Fig. 4):
  1. Intrabyte exposure via bit-shifting:
       x0 = x,  xi = (x_{i-1} << 1) mod 256,  i = 1..7
       x_in = stack([x0, x1, ..., x7], axis=row) -> shape (Ns, 8)
     Each row now holds 8 bytes that differ from each other by a 1-bit
     shift (intrabyte info); each column differs by an 8-bit / 1-byte
     shift (interbyte info, same as the raw sequence).
  2. Intrabyte n-grams: group `n` consecutive rows of x_in together and
     concatenate them along the width axis to widen the image
     (H = Ns - n + 1, W = 8*n), which both raises the (low) aspect ratio
     8/Ns towards something CNNs handle better, and keeps every adjacent
     pair of bytes within a 1-bit shift of each other.
  3. (Training only) light image augmentation — handled in dataset.py so
     that raw byte sequences stay available for the byte branch.

This module is dependency-light (NumPy only) so it can be unit tested and
visualized without importing torch.
"""
from __future__ import annotations

import numpy as np


def bit_shift_stack(byte_seq: np.ndarray) -> np.ndarray:
    """Expose intrabyte information by bit-shifting a byte sequence 7 times.

    Args:
        byte_seq: 1D array of shape (Ns,), dtype uint8 (or any int dtype),
            values in [0, 255] — the raw bytes of one file fragment.

    Returns:
        x_in: array of shape (Ns, 8), dtype uint8. Column j = byte_seq
        left-shifted by j bits (mod 256), matching Eq. (2) in the paper
        (x_i = x_{i-1} << 1).
    """
    byte_seq = np.asarray(byte_seq, dtype=np.uint8)
    ns = byte_seq.shape[0]
    x_in = np.empty((ns, 8), dtype=np.uint8)
    cur = byte_seq.copy()
    for i in range(8):
        x_in[:, i] = cur
        # left shift by 1 bit within a byte (wraps as in the paper's
        # repeated `<< 1` on the 8-bit representation)
        cur = ((cur.astype(np.uint16) << 1) & 0xFF).astype(np.uint8)
    return x_in


def intrabyte_ngram(x_in: np.ndarray, n: int = 4) -> np.ndarray:
    """Widen x_in into a more-square grayscale image using intrabyte n-grams.

    Each row of x_in is treated as a "unigram" (Sec. III-B). An n-gram
    groups n consecutive rows and concatenates them column-wise, giving
    an image of shape (Ns - n + 1, 8*n), consistent with Eq. (3):
        H = Ns - n + 1,   W = 8n

    Args:
        x_in: array of shape (Ns, 8) from bit_shift_stack.
        n: n-gram order (1, 2, 4, 8, 16, ... in the paper's ablation).

    Returns:
        x_ngram: array of shape (Ns - n + 1, 8*n), dtype uint8.
    """
    ns = x_in.shape[0]
    h = ns - n + 1
    if h <= 0:
        raise ValueError(f"n-gram {n} too large for sequence length {ns}")
    # sliding_window_view avoids an explicit Python loop over H rows
    windows = np.lib.stride_tricks.sliding_window_view(x_in, (n, 8))
    # windows shape: (H, 1, n, 8) -> (H, n, 8) -> (H, n*8)
    x_ngram = windows[:, 0, :, :].reshape(h, n * 8)
    return x_ngram


def byte2image(byte_seq: np.ndarray, n: int = 4) -> np.ndarray:
    """Full Byte2Image transform: raw bytes -> 2D grayscale image.

    Args:
        byte_seq: 1D array of shape (Ns,) raw byte values.
        n: intrabyte n-gram order.

    Returns:
        Grayscale image of shape (H, W) = (Ns - n + 1, 8n), values in
        [0, 255] (float32, NOT yet normalized to [0, 1] — the dataset
        loader / augmentation pipeline handles normalization).
    """
    x_in = bit_shift_stack(byte_seq)
    x_ngram = intrabyte_ngram(x_in, n=n)
    return x_ngram.astype(np.float32)


if __name__ == "__main__":
    # tiny smoke test / visualization aid
    rng = np.random.default_rng(0)
    frag = rng.integers(0, 256, size=512, dtype=np.uint8)
    img = byte2image(frag, n=4)
    print("input bytes:", frag.shape, frag.dtype)
    print("Byte2Image output:", img.shape, img.dtype, img.min(), img.max())
    assert img.shape == (512 - 4 + 1, 8 * 4)
    print("OK")
