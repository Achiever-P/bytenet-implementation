"""Generates figures/architecture_diagram.png -- the ByteNet dual-branch
architecture, drawn to match the actual class structure in bytenet/models.py
(ByteBranch, NgramEmbeddingSimple/PatchEmbedding, RVFEStage x4, fusion)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#264653"
TEAL = "#2A9D8F"
SAND = "#E9C46A"
ORANGE = "#E76F51"

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=9.5, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=0, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight="bold", wrap=True)
    return (x, y, w, h)


def arrow(p1, p2, **kw):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.6, color="#555555", **kw)
    ax.add_patch(a)


# Input
box(0.1, 4.3, 1.6, 1.0, "Raw bytes\n(Ns,)", "#888888", fontsize=9)
arrow((1.7, 4.8), (2.15, 4.8))

# Byte branch
box(2.2, 4.2, 2.2, 1.2, "Byte Branch\n(BBFE)\nFC layer", SAND, textcolor=NAVY)
arrow((1.7, 1.5), (2.15, 1.5))

# Byte2Image + image branch
box(0.1, 1.0, 1.6, 1.0, "Byte2Image\n(H, W)", "#888888", fontsize=9)
box(2.2, 0.2, 2.4, 2.6,
    "Image Branch (IBFE)\n\nEmbedding layer\n\u2193\nStage 1 (RVFE+conv)\n\u2193\nStage 2 (RVFE+conv)\n\u2193\nStage 3 (RVFE+conv)\n\u2193\nStage 4 (RVFE+GAP)",
    TEAL, fontsize=8.3)

# Arrows into fusion
arrow((4.4, 4.8), (5.6, 3.4))
arrow((4.6, 1.5), (5.6, 2.9))

# Fusion
box(5.7, 2.5, 2.0, 1.3, "Feature\nFusion\n(concat)", NAVY)

arrow((7.7, 3.15), (8.4, 3.15))

# Classifier
box(8.5, 2.6, 2.3, 1.1, "FC + Softmax", ORANGE)
arrow((10.8, 3.15), (11.6, 3.15))
ax.text(11.65, 3.15, "class\nprobs", ha="left", va="center", fontsize=9)

# Variant note
ax.text(3.4, 0.05, "RVFE block = ResNet block (ByteResNet)  or  PoolFormer block (ByteFormer)",
        ha="center", fontsize=8.5, style="italic", color="#444444")

ax.text(0.1, 5.7, "ByteNet dual-branch architecture (as implemented in bytenet/models.py)",
        fontsize=12.5, weight="bold", color=NAVY)

fig.tight_layout()
fig.savefig("outputs/figures/architecture_diagram.png", dpi=150, bbox_inches="tight")
print("saved outputs/figures/architecture_diagram.png")
