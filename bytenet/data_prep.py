"""
Builds a small, FFT-75-style multimedia file fragment classification
corpus from real files already present on the local filesystem.

Rationale: the original paper evaluates on FFT-75 (75 file types x
102,400 sectors each) and VFF-16, neither of which is downloadable in
this environment. To reproduce the *method* faithfully and end-to-end
(rather than only the architecture on synthetic noise), this script
scans the local disk for real files of several common formats, and
extracts fixed-size byte sectors from random offsets inside them --
exactly how FFT-75 itself was constructed from GovDocs source files.

Classes (chosen to mirror FFT-75's "grouping tags", Table I of the paper):
  py    - source code / plain text            (Text tag)
  json  - structured text                      (Text tag)
  html  - markup                               (Text tag)
  png   - image                                (Bitmap tag)
  pdf   - published document                   (Published tag)
  gz    - compressed archive                   (Archive tag)
  elf   - compiled executable/binary           (Executable tag)

Usage:
    python -m bytenet.data_prep --sector-size 512 --per-class 400 \
        --out data/fragments.npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

CLASS_GLOBS = {
    "py": ["*.py"],
    "json": ["*.json"],
    "html": ["*.html"],
    "png": ["*.png"],
    "pdf": ["*.pdf"],
    "gz": ["*.gz"],
    "elf": None,  # handled specially: executables under /usr/bin
}

SEARCH_ROOTS = ["/usr", "/opt", "/home", "/etc"]
EXCLUDE_DIRS = {"proc", "sys", "dev"}


def _iter_files(ext_globs, min_size):
    seen = set()
    for root in SEARCH_ROOTS:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for pattern in ext_globs:
            for p in rootp.rglob(pattern):
                if any(part in EXCLUDE_DIRS for part in p.parts):
                    continue
                if p in seen or not p.is_file():
                    continue
                try:
                    if p.stat().st_size >= min_size:
                        seen.add(p)
                        yield p
                except OSError:
                    continue


def _iter_elf_files(min_size):
    seen = set()
    for root in ["/usr/bin", "/usr/lib", "/usr/lib/x86_64-linux-gnu"]:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for p in rootp.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if p in seen:
                continue
            try:
                if p.stat().st_size < min_size:
                    continue
                with open(p, "rb") as f:
                    magic = f.read(4)
                if magic[:4] == b"\x7fELF":
                    seen.add(p)
                    yield p
            except OSError:
                continue


def collect_class_files(class_name, sector_size, max_files=4000):
    min_size = sector_size * 2
    if class_name == "elf":
        files = list(_iter_elf_files(min_size))
    else:
        files = list(_iter_files(CLASS_GLOBS[class_name], min_size))
    random.shuffle(files)
    return files[:max_files]


def extract_sectors(files, sector_size, n_sectors, rng):
    """Randomly sample `n_sectors` fixed-size byte windows from `files`."""
    sectors = []
    files = list(files)
    if not files:
        return sectors
    attempts = 0
    max_attempts = n_sectors * 20
    while len(sectors) < n_sectors and attempts < max_attempts:
        attempts += 1
        fpath = files[rng.integers(0, len(files))]
        try:
            size = fpath.stat().st_size
            if size < sector_size:
                continue
            offset = int(rng.integers(0, size - sector_size + 1))
            with open(fpath, "rb") as f:
                f.seek(offset)
                chunk = f.read(sector_size)
            if len(chunk) == sector_size:
                sectors.append(np.frombuffer(chunk, dtype=np.uint8))
        except OSError:
            continue
    return sectors


def build_dataset(sector_size=512, per_class=400, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    class_names = sorted(CLASS_GLOBS.keys())
    X, y = [], []
    report = {}
    for label, cname in enumerate(class_names):
        files = collect_class_files(cname, sector_size)
        sectors = extract_sectors(files, sector_size, per_class, rng)
        report[cname] = {"n_source_files": len(files), "n_sectors": len(sectors)}
        for s in sectors:
            X.append(s)
            y.append(label)

    X = np.stack(X).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    return X, y, class_names, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sector-size", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/fragments.npz")
    args = ap.parse_args()

    X, y, class_names, report = build_dataset(
        sector_size=args.sector_size, per_class=args.per_class, seed=args.seed
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, y=y, class_names=np.array(class_names), sector_size=args.sector_size
    )

    print(f"Saved {X.shape[0]} sectors of size {args.sector_size} bytes -> {args.out}")
    print(f"Classes: {class_names}")
    for cname, info in report.items():
        print(f"  {cname:6s}: {info['n_sectors']:4d} sectors from {info['n_source_files']:4d} files")


if __name__ == "__main__":
    main()
