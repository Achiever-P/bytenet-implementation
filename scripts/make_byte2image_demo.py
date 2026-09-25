"""Generates figures/byte2image_demo.png -- a REAL Byte2Image transform
run through the actual bytenet.byte2image module on a real PNG fragment
pulled from our dataset, so this is evidence of the implementation
running, not an illustration."""
import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bytenet.byte2image import bit_shift_stack, intrabyte_ngram, byte2image

data = np.load("data/fragments.npz", allow_pickle=True)
X, y, class_names = data["X"], data["y"], list(data["class_names"])

# grab one real PNG fragment
png_label = class_names.index("png")
frag = X[y == png_label][3]  # pick one

x_in = bit_shift_stack(frag)              # (256, 8)
img4 = intrabyte_ngram(x_in, n=4)         # (253, 32)
img16 = intrabyte_ngram(x_in, n=16)       # (241, 128)

fig = plt.figure(figsize=(11, 5.2))
gs = fig.add_gridspec(1, 4, width_ratios=[0.6, 1.1, 1.6, 2.8], wspace=0.35)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(frag.reshape(-1, 1), cmap="gray", aspect="auto", vmin=0, vmax=255)
ax0.set_title("Raw bytes\n(256,)", fontsize=10)
ax0.set_xticks([])
ax0.set_ylabel("byte index")

ax1 = fig.add_subplot(gs[1])
ax1.imshow(x_in, cmap="gray", aspect="auto", vmin=0, vmax=255)
ax1.set_title("After bit-shift\n(256, 8)", fontsize=10)
ax1.set_xlabel("shift 0..7")
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[2])
ax2.imshow(img4, cmap="gray", aspect="auto", vmin=0, vmax=255)
ax2.set_title("n-gram=4\n(253, 32)", fontsize=10)
ax2.set_xlabel("width")
ax2.set_yticks([])

ax3 = fig.add_subplot(gs[3])
ax3.imshow(img16, cmap="gray", aspect="auto", vmin=0, vmax=255)
ax3.set_title("n-gram=16 (paper's default setting)\n(241, 128)", fontsize=10)
ax3.set_xlabel("width")
ax3.set_yticks([])

fig.suptitle("Byte2Image transform running on a real 256-byte PNG fragment from our dataset", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("outputs/figures/byte2image_demo.png", dpi=150, bbox_inches="tight")
print("saved outputs/figures/byte2image_demo.png")
print("raw bytes sample:", frag[:12].tolist())
