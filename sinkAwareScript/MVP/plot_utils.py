from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "baseline_prompt_only": "#7f8c8d",
    "first_k": "#2c7fb8",
    "sink_top_b": "#31a354",
    "sink_entropy_top_b": "#756bb1",
    "random": "#fdae6b",
    "same_bucket_random": "#e6550d",
}


def save_bar(values: dict[str, float], title: str, ylabel: str, path: Path) -> None:
    if not values:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return

    names = list(values.keys())
    heights = [values[name] for name in names]
    colors = [COLORS.get(name, "#999999") for name in names]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(names, heights, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(heights):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(xs, ys, colors, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.75, s=28, edgecolors="none")
    ax.axhline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.axvline(0, color="#444444", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hist(values, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=20, color="#31a354", alpha=0.85)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
