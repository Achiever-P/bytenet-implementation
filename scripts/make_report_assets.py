"""Turns outputs/*/result.json into the figures and tables used in the
report and slides: training curves, ablation bar chart, n-gram sweep
chart, and a confusion matrix heatmap for the main ByteResNet run.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(OUT / name / "result.json") as f:
        return json.load(f)


def fig_training_curves():
    r_resnet = load("resnet_full")
    r_former = load("poolformer_full")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(r_resnet["history"]["train_acc"], label="ByteResNet train", marker="o", ms=3)
    ax.plot(r_resnet["history"]["test_acc"], label="ByteResNet test", marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("ByteResNet training curve (n-gram=4)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(r_former["history"]["train_acc"], label="ByteFormer train", marker="o", ms=3, color="tab:green")
    ax.plot(r_former["history"]["test_acc"], label="ByteFormer test", marker="o", ms=3, color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("ByteFormer training curve (n-gram=4)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=150)
    plt.close(fig)


def fig_confusion_matrix():
    r = load("resnet_full")
    conf = np.array(r["confusion_matrix"])
    conf_norm = conf / conf.sum(axis=1, keepdims=True)
    names = r["class_names"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"ByteResNet confusion matrix (test acc={r['final_test_acc']:.1%})")
    for i in range(len(names)):
        for j in range(len(names)):
            val = conf_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def fig_ablation_bars():
    full = load("resnet_full")["final_test_acc"]
    byte_only = load("resnet_byte_only")["final_test_acc"]
    image_only = load("resnet_image_only")["final_test_acc"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["Byte branch\nonly", "Image branch\nonly", "Full\n(dual-branch)"]
    vals = [byte_only, image_only, full]
    colors = ["#f4a261", "#2a9d8f", "#264653"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Branch ablation (ByteResNet, n-gram=4)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_branches.png", dpi=150)
    plt.close(fig)


def fig_ngram_sweep():
    ns = [2, 4, 8, 16]
    accs = []
    for n in ns:
        name = "resnet_full" if n == 4 else f"resnet_ngram{n}"
        accs.append(load(name)["final_test_acc"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ns, accs, marker="o", ms=8, color="tab:purple")
    for n, a in zip(ns, accs):
        ax.annotate(f"{a:.1%}", (n, a), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("Intrabyte n-gram order")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of n-gram order (ByteResNet)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ngram_sweep.png", dpi=150)
    plt.close(fig)


def fig_variant_comparison():
    r_resnet = load("resnet_full")
    r_former = load("poolformer_full")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["ByteResNet", "ByteFormer"]
    accs = [r_resnet["final_test_acc"], r_former["final_test_acc"]]
    params = [r_resnet["n_params"], r_former["n_params"]]
    colors = ["#264653", "#e76f51"]
    bars = ax.bar(names, accs, color=colors)
    for b, v, p in zip(bars, accs, params):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}\n({p/1e3:.0f}K params)",
                ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("ByteResNet vs. ByteFormer")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "variant_comparison.png", dpi=150)
    plt.close(fig)


def print_summary_table():
    rows = []
    configs = [
        ("ByteResNet (full)", "resnet_full"),
        ("ByteFormer (full)", "poolformer_full"),
        ("ByteResNet (byte branch only)", "resnet_byte_only"),
        ("ByteResNet (image branch only)", "resnet_image_only"),
        ("ByteResNet (n-gram=2)", "resnet_ngram2"),
        ("ByteResNet (n-gram=8)", "resnet_ngram8"),
        ("ByteResNet (n-gram=16)", "resnet_ngram16"),
    ]
    print(f"{'Configuration':35s} {'Test Acc':>10s} {'Best Acc':>10s} {'Params':>10s}")
    print("-" * 68)
    for label, name in configs:
        r = load(name)
        rows.append((label, r["final_test_acc"], r["best_test_acc"], r["n_params"]))
        print(f"{label:35s} {r['final_test_acc']:>9.1%} {r['best_test_acc']:>9.1%} {r['n_params']:>10,}")

    with open(OUT / "summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    fig_training_curves()
    fig_confusion_matrix()
    fig_ablation_bars()
    fig_ngram_sweep()
    fig_variant_comparison()
    print_summary_table()
    print(f"\nFigures saved to {FIG}/")
