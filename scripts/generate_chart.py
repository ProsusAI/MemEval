import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

systems = {
    "PropMem (Ours)": (5_900_000 / 1986, 0.605, "*", "#1d4ed8", 600),
    "OpenClaw":       (16_400_000 / 1986, 0.557, "P", "#d35400", 110),
    "Full Context":   (37_500_000 / 1986, 0.542, "X", "#e05c5c", 120),
    "Hindsight":      (24_200_000 / 1986, 0.489, "h", "#f39c12", 110),
    "Graphiti":       (4_800_000 / 1986, 0.416, "^", "#27ae60", 110),
    "SimpleMem":      (11_400_000 / 1986, 0.358, "s", "#4a90d9", 110),
    "Mem0":           (3_000_000 / 1986, 0.344, "D", "#8e44ad", 110),
    "MemU":           (6_700_000 / 1986, 0.299, "v", "#7f8c8d", 110),
}

star_x = 5_900_000 / 1986
star_y = 0.605

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#f7f9fc")

# Draw star glow layers (largest to smallest, most to least transparent)
for size, alpha in [(2500, 0.07), (1600, 0.12), (1000, 0.18)]:
    ax.scatter(
        star_x,
        star_y,
        marker="o",
        color="#1d4ed8",
        s=size,
        alpha=alpha,
        zorder=6,
        linewidths=0,
    )

# White backing star (slightly larger) for the white-edge effect
ax.scatter(star_x, star_y, marker="*", color="white", s=750, zorder=7, linewidths=0)

# Main blue star
ax.scatter(
    star_x,
    star_y,
    marker="*",
    color="#1d4ed8",
    s=580,
    zorder=8,
    edgecolors="white",
    linewidths=1.2,
)

# All other systems
for name, (tokens, f1, marker, color, size) in systems.items():
    if name == "PropMem (Ours)":
        continue
    ax.scatter(
        tokens,
        f1,
        marker=marker,
        color=color,
        s=size,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )

# Callout box
callout = (
    "  PropMem Advantages\n"
    "  ✔  #1 Overall F1 (0.605)\n"
    "  ✔  6.4x fewer tokens than Full Context"
)
ax.annotate(
    callout,
    xy=(0.02, 0.97),
    xycoords="axes fraction",
    fontsize=8.5,
    va="top",
    ha="left",
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="#e8f4fd",
        edgecolor="#1d4ed8",
        linewidth=2,
    ),
)

ax.set_xscale("log")
ax.set_xlabel("Average Token Cost per Question (log scale)", fontsize=10)
ax.set_ylabel("Performance (F1 Score)", fontsize=10)
ax.set_title(
    "Performance vs Cost — LoCoMo Benchmark (gpt-4.1-mini)",
    fontsize=11,
    fontweight="bold",
)

ax.set_xlim(200, 30000)
ax.set_ylim(0.22, 0.65)

ax.grid(True, which="both", linestyle="--", alpha=0.35, color="#aaaaaa")
ax.spines[["top", "right"]].set_visible(False)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker=m,
        color="w",
        markerfacecolor=c,
        markersize=12 if m == "*" else 8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=n,
    )
    for n, (_, _, m, c, _) in systems.items()
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=8.5,
    framealpha=0.95,
    edgecolor="#cccccc",
    title="Systems",
    title_fontsize=9,
)

plt.tight_layout()
plt.savefig("assets/benchmark.png", dpi=150, bbox_inches="tight")
print("Saved assets/benchmark.png")