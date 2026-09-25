"""
Training script for ByteNet (ByteResNet / ByteFormer variants).

Mirrors the paper's training recipe (Sec. IV-B) at reduced scale:
  - AdamW, weight decay 0.01
  - cosine LR schedule with linear warmup
  - negative log-likelihood loss on (mixup-softened) labels, Eq. (19)

Usage:
    python -m bytenet.train --data data/fragments.npz --variant resnet \
        --epochs 12 --n-gram 4 --out outputs/resnet_full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bytenet.dataset import FragmentDataset, mixup
from bytenet.models import ByteNet


def soft_nll_loss(logits, soft_targets):
    """Eq. (19): NLL loss against a (possibly mixup-softened) label dist."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def split_train_test(X, y, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    X, y = data["X"], data["y"]
    class_names = list(data["class_names"])
    sector_size = int(data["sector_size"])
    num_classes = len(class_names)

    Xtr, ytr, Xte, yte = split_train_test(X, y, test_frac=args.test_frac, seed=args.seed)

    train_ds = FragmentDataset(Xtr, ytr, n_gram=args.n_gram, train=True)
    test_ds = FragmentDataset(Xte, yte, n_gram=args.n_gram, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    h = sector_size - args.n_gram + 1
    w = 8 * args.n_gram

    model = ByteNet(
        sector_size=sector_size,
        img_h=h,
        img_w=w,
        num_classes=num_classes,
        variant=args.variant,
        stage_channels=tuple(args.channels),
        stage_depths=tuple(args.depths),
        ngram_k=args.ngram_k,
        patch_size=args.patch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.999))
    warmup_epochs = max(1, args.epochs // 6)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_time_s": []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for raw, img, labels in train_loader:
            optimizer.zero_grad()

            if args.ablation == "byte_only":
                img = torch.zeros_like(img)
            elif args.ablation == "image_only":
                raw = torch.zeros_like(raw)

            if args.mixup_alpha > 0:
                raw_m, img_m, y_soft = mixup(raw, img, labels, num_classes, alpha=args.mixup_alpha)
                logits = model(raw_m, img_m)
                loss = soft_nll_loss(logits, y_soft)
            else:
                logits = model(raw, img)
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * raw.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += raw.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = loss_sum / total

        # ---- eval ----
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for raw, img, labels in test_loader:
                if args.ablation == "byte_only":
                    img = torch.zeros_like(img)
                elif args.ablation == "image_only":
                    raw = torch.zeros_like(raw)
                logits = model(raw, img)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += raw.size(0)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        test_acc = correct / total
        best_acc = max(best_acc, test_acc)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time_s"].append(dt)

        print(f"[{args.variant}/{args.ablation}/n{args.n_gram}] "
              f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  ({dt:.1f}s)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # confusion matrix on final epoch predictions
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(all_preds, all_labels):
        conf[t, p] += 1

    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "variant": args.variant,
        "ablation": args.ablation,
        "n_gram": args.n_gram,
        "sector_size": sector_size,
        "class_names": class_names,
        "history": history,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "confusion_matrix": conf.tolist(),
        "n_params": n_params,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "args": vars(args),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved results to {out_dir}")
    return result


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/fragments.npz")
    ap.add_argument("--variant", type=str, default="resnet", choices=["resnet", "poolformer"])
    ap.add_argument("--ablation", type=str, default="full",
                     choices=["full", "byte_only", "image_only"])
    ap.add_argument("--n-gram", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mixup-alpha", type=float, default=0.8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--channels", type=int, nargs=4, default=[32, 64, 128, 256])
    ap.add_argument("--depths", type=int, nargs=4, default=[1, 1, 1, 1])
    ap.add_argument("--ngram-k", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--out", type=str, default="outputs/run")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
