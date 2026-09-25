"""Runs the full experiment matrix used for the report/slides:

  1. ByteResNet (full, dual-branch)      -- main result
  2. ByteFormer (full, dual-branch)      -- main result / variant comparison
  3. ByteResNet, byte-branch only        -- ablation (Table V style)
  4. ByteResNet, image-branch only       -- ablation (Table V style)
  5. ByteResNet across n-gram in {2,4,8,16} -- ablation (Fig. 11 style)

Results land in outputs/<name>/result.json ; run scripts/make_report_assets.py
afterwards to turn them into the plots/tables used in the report and slides.
"""
import subprocess
import sys

RUNS = [
    dict(name="resnet_full", variant="resnet", ablation="full", n_gram=4, epochs=16),
    dict(name="poolformer_full", variant="poolformer", ablation="full", n_gram=4, epochs=16),
    dict(name="resnet_byte_only", variant="resnet", ablation="byte_only", n_gram=4, epochs=10),
    dict(name="resnet_image_only", variant="resnet", ablation="image_only", n_gram=4, epochs=10),
    dict(name="resnet_ngram2", variant="resnet", ablation="full", n_gram=2, epochs=10),
    dict(name="resnet_ngram8", variant="resnet", ablation="full", n_gram=8, epochs=10),
    dict(name="resnet_ngram16", variant="resnet", ablation="full", n_gram=16, epochs=10),
]

BASE_CMD = [
    sys.executable, "-m", "bytenet.train",
    "--data", "data/fragments.npz",
    "--batch-size", "64",
    "--channels", "16", "32", "64", "128",
]


def main():
    for run in RUNS:
        cmd = BASE_CMD + [
            "--variant", run["variant"],
            "--ablation", run["ablation"],
            "--n-gram", str(run["n_gram"]),
            "--epochs", str(run["epochs"]),
            "--out", f"outputs/{run['name']}",
        ]
        print("=" * 70)
        print("RUNNING:", " ".join(cmd))
        print("=" * 70)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
