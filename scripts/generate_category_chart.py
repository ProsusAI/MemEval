import matplotlib.pyplot as plt
import numpy as np


def generate_bar_chart(
    title, categories, systems, output_path, figsize=(12, 5.5), bar_width=0.085,
    legend_y=-0.08,
):
    n_systems = len(systems)
    x = np.arange(len(categories))
    total_width = bar_width * n_systems

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f9fc")

    for i, (name, (scores, color)) in enumerate(systems.items()):
        offset = x - total_width / 2 + bar_width * (i + 0.5)
        is_propmem = name == "PropMem (Ours)"
        ax.bar(
            offset,
            scores,
            bar_width,
            label=name,
            color=color,
            edgecolor="#0f3299" if is_propmem else "white",
            linewidth=1.5 if is_propmem else 0.5,
            alpha=1.0 if is_propmem else 0.75,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Token F1 Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(s for _, (scores, _) in systems.items() for s in scores) + 0.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize=8,
        framealpha=0.95,
        edgecolor="#cccccc",
        title="Systems",
        title_fontsize=9,
        ncol=min(n_systems, 5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# LoCoMo per-category F1
generate_bar_chart(
    title="Per-Category F1 — LoCoMo Benchmark (gpt-4.1-mini)",
    categories=["Factual", "Temporal", "Multi-hop", "Inferential", "Adversarial"],
    systems={
        "PropMem (Ours)": ([0.431, 0.615, 0.599, 0.289, 0.794], "#1d4ed8"),
        "OpenClaw":       ([0.464, 0.482, 0.670, 0.213, 0.528], "#d35400"),
        "Full Context":   ([0.517, 0.369, 0.674, 0.197, 0.509], "#e05c5c"),
        "Hindsight":      ([0.431, 0.306, 0.526, 0.206, 0.647], "#f39c12"),
        "Graphiti":       ([0.296, 0.151, 0.349, 0.120, 0.873], "#27ae60"),
        "Memory-R1":      ([0.370, 0.116, 0.460, 0.193, 0.504], "#14b8a6"),
        "SimpleMem":      ([0.245, 0.320, 0.237, 0.136, 0.734], "#4a90d9"),
        "Mem0":           ([0.267, 0.104, 0.330, 0.174, 0.629], "#8e44ad"),
        "MemU":           ([0.190, 0.068, 0.233, 0.076, 0.704], "#7f8c8d"),
    },
    output_path="assets/benchmark_categories.png",
)

# LongMemEval per-category F1
generate_bar_chart(
    title="Per-Category F1 — LongMemEval Benchmark (gpt-4.1)",
    categories=[
        "Single-Session\nUser",
        "Single-Session\nAssistant",
        "Single-Session\nPreference",
        "Multi-\nSession",
        "Temporal",
        "Knowledge\nUpdate",
    ],
    systems={
        "PropMem (Ours)": ([0.851, 0.767, 0.147, 0.582, 0.424, 0.528], "#1d4ed8"),
        "SimpleMem":      ([0.752, 0.566, 0.126, 0.382, 0.578, 0.475], "#4a90d9"),
        "OpenClaw":       ([0.401, 0.432, 0.127, 0.082, 0.185, 0.234], "#d35400"),
        "Full Context":   ([0.265, 0.415, 0.177, 0.062, 0.212, 0.202], "#e05c5c"),
    },
    output_path="assets/benchmark_longmemeval_categories.png",
    figsize=(11, 5.5),
    bar_width=0.15,
    legend_y=-0.18,
)
