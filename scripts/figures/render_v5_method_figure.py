#!/usr/bin/env python3
"""Render the publication figure for the three-stage MemGen V5 method."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

CACHE_ROOT = Path(tempfile.gettempdir()) / "memgen-figure-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "memgen" / "figures"
STEM = OUTPUT_DIR / "v5_three_stage_method"

INK = "#18212B"
MUTED = "#5B6573"
LINE = "#8A95A3"
PANEL_EDGE = "#C8D0DA"
WHITE = "#FFFFFF"

BLUE_FILL = "#E8F2FB"
BLUE_EDGE = "#4C86B6"
AMBER_FILL = "#FFF3D7"
AMBER_EDGE = "#C9902F"
GREEN_FILL = "#E7F5EC"
GREEN_EDGE = "#4E9B68"
PURPLE_FILL = "#F0EAFB"
PURPLE_EDGE = "#8365B2"
GRAY_FILL = "#F4F6F8"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"],
            "font.size": 8.2,
            "text.color": INK,
            "axes.edgecolor": LINE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    weight: str = "normal",
    fontsize: float = 8.0,
    linewidth: float = 1.0,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.075",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.15,
        zorder=4,
    )
    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "center": (x + width / 2, y + height / 2),
    }


def arrow(
    ax,
    start,
    end,
    *,
    color: str = LINE,
    dashed: bool = False,
    linewidth: float = 1.05,
    mutation_scale: float = 9.0,
    connectionstyle: str = "arc3,rad=0",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=(0, (3.0, 2.2)) if dashed else "solid",
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1.2,
        shrinkB=1.2,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def orthogonal_arrow(ax, points, *, color=LINE, dashed=True, linewidth=1.0):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    path = MplPath(points, codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle=(0, (3.0, 2.2)) if dashed else "solid",
            zorder=1,
        )
    )
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        dashed=dashed,
        linewidth=linewidth,
        mutation_scale=8.0,
    )


def panel(ax, x, y, width, height, label, title, split, tint, accent):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=tint,
        edgecolor=PANEL_EDGE,
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.20,
        y + height - 0.28,
        label,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x + 0.58,
        y + height - 0.28,
        title,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + width - 0.18,
        y + height - 0.28,
        split.upper(),
        ha="right",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=accent,
    )


def draw_legend(ax):
    items = [
        (BLUE_FILL, BLUE_EDGE, "冻结模型"),
        (AMBER_FILL, AMBER_EDGE, "教师辅助"),
        (GREEN_FILL, GREEN_EDGE, "持久化产物"),
        (PURPLE_FILL, PURPLE_EDGE, "校准策略"),
    ]
    x = 4.50
    y = 5.98
    for fill, edge, label in items:
        ax.add_patch(
            Rectangle(
                (x, y - 0.075),
                0.15,
                0.15,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.9,
                zorder=5,
            )
        )
        ax.text(x + 0.21, y, label, ha="left", va="center", fontsize=7.0, color=MUTED)
        x += 1.80 if label != "持久化产物" else 2.05


def draw_figure():
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(15.2, 6.2), constrained_layout=False)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 6.15)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_legend(ax)

    panel(ax, 0.20, 0.52, 5.15, 5.20, "(a)", "记忆库构造", "仅训练集", "#F8FBFE", BLUE_EDGE)
    panel(ax, 5.65, 0.52, 4.25, 5.20, "(b)", "选择器校准", "仅验证集", "#FCFAFE", PURPLE_EDGE)
    panel(ax, 10.20, 0.52, 4.80, 5.20, "(c)", "在线记忆使用", "测试 / 部署", "#F8FCF9", GREEN_EDGE)

    # Stage I: offline memory construction.
    a_x, a_w, a_h = 0.62, 4.30, 0.48
    a_y = [4.76, 4.10, 3.44, 2.78, 2.12, 1.46, 0.80]
    a_nodes = [
        rounded_box(ax, a_x, a_y[0], a_w, a_h, "训练输入", facecolor=WHITE),
        rounded_box(ax, a_x, a_y[1], a_w, a_h, "冻结推理模型：1 条贪心 + 7 条采样", facecolor=BLUE_FILL, edgecolor=BLUE_EDGE),
        rounded_box(ax, a_x, a_y[2], a_w, a_h, "轨迹审查与同题成功—失败对照", facecolor=AMBER_FILL, edgecolor=AMBER_EDGE),
        rounded_box(ax, a_x, a_y[3], a_w, a_h, "经验原子提取", facecolor=AMBER_FILL, edgecolor=AMBER_EDGE),
        rounded_box(ax, a_x, a_y[4], a_w, a_h, "基于共同方法的语义分组", facecolor=AMBER_FILL, edgecolor=AMBER_EDGE),
        rounded_box(ax, a_x, a_y[5], a_w, a_h, "记忆卡构造与资格审查", facecolor=AMBER_FILL, edgecolor=AMBER_EDGE),
        rounded_box(ax, a_x, a_y[6], a_w, a_h, "主级（Primary）记忆库：选择元数据 + 原生前缀 KV", facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, weight="bold"),
    ]
    for upper, lower in zip(a_nodes, a_nodes[1:]):
        arrow(ax, upper["bottom"], lower["top"])

    # Stage II: valid-only calibration.
    b_x, b_w, b_h = 6.03, 3.48, 0.58
    b_y = [4.62, 3.70, 2.78, 1.86, 0.86]
    b_nodes = [
        rounded_box(ax, b_x, b_y[0], b_w, b_h, "验证输入", facecolor=WHITE),
        rounded_box(ax, b_x, b_y[1], b_w, b_h, "语义 Top-K + 排除条件感知重排", facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE),
        rounded_box(ax, b_x, b_y[2], b_w, b_h, "无记忆基线与候选 KV 配对执行", facecolor=BLUE_FILL, edgecolor=BLUE_EDGE),
        rounded_box(ax, b_x, b_y[3], b_w, b_h, "配对效用标签：增益、伤害、中性", facecolor=WHITE),
        rounded_box(ax, b_x, b_y[4], b_w, b_h, "岭回归效用模型 + 弃权阈值", facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE, weight="bold"),
    ]
    for upper, lower in zip(b_nodes, b_nodes[1:]):
        arrow(ax, upper["bottom"], lower["top"])

    # Stage III: frozen online policy.
    c_x, c_w, c_h = 10.67, 3.86, 0.54
    c_y = [4.66, 3.82, 2.98]
    c_nodes = [
        rounded_box(ax, c_x, c_y[0], c_w, c_h, "新输入", facecolor=WHITE),
        rounded_box(ax, c_x, c_y[1], c_w, c_h, "冻结语义检索器 + 适用性重排器", facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE),
        rounded_box(ax, c_x, c_y[2], c_w, c_h, "期望效用评分", facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE),
    ]
    for upper, lower in zip(c_nodes, c_nodes[1:]):
        arrow(ax, upper["bottom"], lower["top"])

    diamond_center = (12.60, 2.23)
    diamond_w, diamond_h = 2.30, 0.72
    diamond = Polygon(
        [
            (diamond_center[0], diamond_center[1] + diamond_h / 2),
            (diamond_center[0] + diamond_w / 2, diamond_center[1]),
            (diamond_center[0], diamond_center[1] - diamond_h / 2),
            (diamond_center[0] - diamond_w / 2, diamond_center[1]),
        ],
        closed=True,
        facecolor=WHITE,
        edgecolor=PURPLE_EDGE,
        linewidth=1.1,
        zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(*diamond_center, "最高效用 > 阈值？", ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=4)
    arrow(ax, c_nodes[-1]["bottom"], (diamond_center[0], diamond_center[1] + diamond_h / 2))

    prefix = rounded_box(ax, 10.48, 1.18, 2.08, 0.52, "选中的原生前缀 KV", facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, weight="bold", fontsize=7.1)
    no_memory = rounded_box(ax, 13.12, 1.18, 1.32, 0.52, "不使用记忆", facecolor=GRAY_FILL, edgecolor=LINE, fontsize=7.4)
    reasoner = rounded_box(ax, 10.72, 0.58, 2.68, 0.46, "冻结推理模型", facecolor=BLUE_FILL, edgecolor=BLUE_EDGE, weight="bold")
    output = rounded_box(ax, 13.72, 0.58, 1.02, 0.46, "模型输出", facecolor=WHITE, fontsize=7.5)

    arrow(ax, (diamond_center[0] - 0.58, diamond_center[1] - 0.19), prefix["top"], color=GREEN_EDGE, connectionstyle="arc3,rad=0.06")
    ax.text(11.55, 1.90, "是", ha="center", va="center", fontsize=6.8, color=GREEN_EDGE, fontweight="bold")
    arrow(ax, (diamond_center[0] + 0.58, diamond_center[1] - 0.19), no_memory["top"], color=LINE, connectionstyle="arc3,rad=-0.06")
    ax.text(13.66, 1.90, "否", ha="center", va="center", fontsize=6.8, color=MUTED, fontweight="bold")
    arrow(ax, prefix["bottom"], (11.42, reasoner["top"][1]), color=GREEN_EDGE)
    arrow(ax, no_memory["bottom"], (12.88, reasoner["top"][1]), color=LINE, connectionstyle="arc3,rad=-0.04")
    arrow(ax, reasoner["right"], output["left"], color=BLUE_EDGE)

    # Cross-stage frozen artifacts. Orthogonal routing keeps the main flow readable.
    orthogonal_arrow(
        ax,
        [a_nodes[-1]["right"], (5.50, a_nodes[-1]["right"][1]), (5.50, b_nodes[1]["left"][1]), b_nodes[1]["left"]],
        color=GREEN_EDGE,
    )
    ax.text(5.50, 2.44, "记忆库", rotation=90, ha="center", va="center", fontsize=6.3, color=GREEN_EDGE, fontweight="bold")

    orthogonal_arrow(
        ax,
        [b_nodes[-1]["right"], (10.05, b_nodes[-1]["right"][1]), (10.05, c_nodes[1]["left"][1]), c_nodes[1]["left"]],
        color=PURPLE_EDGE,
    )
    ax.text(10.05, 2.37, "策略", rotation=90, ha="center", va="center", fontsize=6.3, color=PURPLE_EDGE, fontweight="bold")

    # The selected online KV is loaded from the frozen Primary Bank.
    orthogonal_arrow(
        ax,
        [a_nodes[-1]["bottom"], (a_nodes[-1]["bottom"][0], 0.24), (prefix["bottom"][0], 0.24), prefix["bottom"]],
        color=GREEN_EDGE,
    )
    ax.text(7.67, 0.15, "冻结的主级记忆库", ha="center", va="center", fontsize=6.6, color=GREEN_EDGE, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(STEM.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
