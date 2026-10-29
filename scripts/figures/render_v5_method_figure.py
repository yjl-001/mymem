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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
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


def node_anchors(x: float, y: float, width: float, height: float):
    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "center": (x + width / 2, y + height / 2),
    }


def parallelogram_node(ax, x, y, width, height, text, *, facecolor=WHITE, edgecolor=LINE, fontsize=8.0):
    skew = min(0.20, width * 0.08)
    patch = Polygon(
        [(x + skew, y), (x + width, y), (x + width - skew, y + height), (x, y + height)],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return node_anchors(x, y, width, height)


def stacked_rollouts(ax, x, y, width, height):
    for offset, alpha in ((0.14, 0.55), (0.07, 0.75)):
        ax.add_patch(
            FancyBboxPatch(
                (x + offset, y + offset),
                width,
                height,
                boxstyle="round,pad=0.025,rounding_size=0.07",
                facecolor=BLUE_FILL,
                edgecolor=BLUE_EDGE,
                linewidth=0.8,
                alpha=alpha,
                zorder=1,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            facecolor=BLUE_FILL,
            edgecolor=BLUE_EDGE,
            linewidth=1.1,
            zorder=3,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        "冻结推理模型\n1 条贪心 + 7 条采样",
        ha="center",
        va="center",
        fontsize=7.8,
        linespacing=1.18,
        zorder=4,
    )
    return node_anchors(x, y, width + 0.14, height + 0.14)


def contrast_node(ax, x, y, width, height):
    lane_width = width * 0.40
    lane_height = height * 0.37
    success = rounded_box(
        ax,
        x,
        y + height - lane_height,
        lane_width,
        lane_height,
        "成功轨迹",
        facecolor="#E8F3FA",
        edgecolor=BLUE_EDGE,
        fontsize=7.1,
    )
    failure = rounded_box(
        ax,
        x,
        y,
        lane_width,
        lane_height,
        "失败轨迹",
        facecolor=AMBER_FILL,
        edgecolor=AMBER_EDGE,
        fontsize=7.1,
    )
    merge = Circle((x + width * 0.57, y + height / 2), radius=0.115, facecolor=WHITE, edgecolor=AMBER_EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(merge)
    ax.text(x + width * 0.57, y + height / 2, "↔", ha="center", va="center", fontsize=7.0, color=AMBER_EDGE, zorder=4)
    ax.plot([success["right"][0], x + width * 0.57 - 0.115], [success["right"][1], y + height / 2], color=AMBER_EDGE, linewidth=0.9, zorder=2)
    ax.plot([failure["right"][0], x + width * 0.57 - 0.115], [failure["right"][1], y + height / 2], color=AMBER_EDGE, linewidth=0.9, zorder=2)
    ax.text(
        x + width * 0.79,
        y + height / 2,
        "轨迹审查与\n同题对照",
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.18,
        color=INK,
        zorder=4,
    )
    return node_anchors(x, y, width, height)


def hexagon_node(ax, x, y, width, height, text, *, facecolor, edgecolor, fontsize=7.7, weight="normal"):
    inset = min(width * 0.13, 0.35)
    patch = Polygon(
        [
            (x + inset, y),
            (x + width - inset, y),
            (x + width, y + height / 2),
            (x + width - inset, y + height),
            (x + inset, y + height),
            (x, y + height / 2),
        ],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
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
        linespacing=1.15,
        zorder=4,
    )
    return node_anchors(x, y, width, height)


def cluster_node(ax, x, y, width, height):
    center_x = x + width * 0.30
    center_y = y + height * 0.52
    points = [
        (center_x - 0.34, center_y + 0.13),
        (center_x + 0.03, center_y + 0.25),
        (center_x + 0.30, center_y - 0.03),
        (center_x - 0.10, center_y - 0.22),
    ]
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 0), (1, 3)):
        ax.plot(
            [points[left][0], points[right][0]],
            [points[left][1], points[right][1]],
            color=AMBER_EDGE,
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )
    for index, point in enumerate(points):
        ax.add_patch(
            Circle(
                point,
                radius=0.095 if index else 0.115,
                facecolor=AMBER_FILL,
                edgecolor=AMBER_EDGE,
                linewidth=1.0,
                zorder=3,
            )
        )
    ax.text(
        x + width * 0.69,
        y + height / 2,
        "基于共同方法的\n语义分组",
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.18,
        zorder=4,
    )
    return node_anchors(x, y, width, height)


def document_node(ax, x, y, width, height, text, *, facecolor=AMBER_FILL, edgecolor=AMBER_EDGE, fontsize=7.6):
    fold = min(0.22, width * 0.10, height * 0.42)
    patch = Polygon(
        [
            (x, y),
            (x + width, y),
            (x + width, y + height - fold),
            (x + width - fold, y + height),
            (x, y + height),
        ],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.plot(
        [x + width - fold, x + width - fold, x + width],
        [y + height, y + height - fold, y + height - fold],
        color=edgecolor,
        linewidth=0.9,
        zorder=4,
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return node_anchors(x, y, width, height)


def cylinder_node(ax, x, y, width, height, text, *, facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, fontsize=7.5, weight="normal"):
    cap_height = min(0.18, height * 0.30)
    body = Rectangle((x, y + cap_height / 2), width, height - cap_height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.05, zorder=3)
    bottom = Ellipse((x + width / 2, y + cap_height / 2), width, cap_height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.05, zorder=3)
    top = Ellipse((x + width / 2, y + height - cap_height / 2), width, cap_height, facecolor="#F5FBF7", edgecolor=edgecolor, linewidth=1.05, zorder=4)
    ax.add_patch(body)
    ax.add_patch(bottom)
    ax.add_patch(top)
    ax.text(
        x + width / 2,
        y + height / 2 - 0.01,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
        zorder=5,
    )
    return node_anchors(x, y, width, height)


def funnel_node(ax, x, y, width, height, text, *, facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE, fontsize=7.5):
    neck = width * 0.23
    patch = Polygon(
        [
            (x, y + height),
            (x + width, y + height),
            (x + width * 0.62, y + height * 0.43),
            (x + width / 2 + neck / 2, y + height * 0.43),
            (x + width / 2 + neck / 2, y),
            (x + width / 2 - neck / 2, y),
            (x + width / 2 - neck / 2, y + height * 0.43),
            (x + width * 0.38, y + height * 0.43),
        ],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.69,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
        zorder=4,
    )
    return node_anchors(x, y, width, height)


def paired_execution_node(ax, x, y, width, height):
    gap = 0.16
    half = (width - gap) / 2
    rounded_box(ax, x, y, half, height, "无记忆基线", facecolor=BLUE_FILL, edgecolor=BLUE_EDGE, fontsize=7.1)
    rounded_box(ax, x + half + gap, y, half, height, "候选 KV", facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, fontsize=7.1)
    ax.add_patch(
        Circle(
            (x + width / 2, y + height / 2),
            radius=0.105,
            facecolor=WHITE,
            edgecolor=LINE,
            linewidth=0.9,
            zorder=5,
        )
    )
    ax.text(x + width / 2, y + height / 2, "↔", ha="center", va="center", fontsize=6.5, color=MUTED, zorder=6)
    return node_anchors(x, y, width, height)


def utility_labels_node(ax, x, y, width, height):
    labels = [("+1", "增益", GREEN_FILL, GREEN_EDGE), ("−1", "伤害", AMBER_FILL, AMBER_EDGE), ("0", "中性", GRAY_FILL, LINE)]
    centers = [x + width * 0.22, x + width * 0.50, x + width * 0.78]
    for center, (value, label, fill, edge) in zip(centers, labels):
        ax.add_patch(Circle((center, y + height * 0.62), radius=0.17, facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=3))
        ax.text(center, y + height * 0.62, value, ha="center", va="center", fontsize=7.4, fontweight="bold", zorder=4)
        ax.text(center, y + height * 0.16, label, ha="center", va="center", fontsize=6.6, color=MUTED, zorder=4)
    return node_anchors(x, y, width, height)


def score_node(ax, center_x, center_y, radius):
    ax.add_patch(Circle((center_x, center_y), radius=radius, facecolor=PURPLE_FILL, edgecolor=PURPLE_EDGE, linewidth=1.1, zorder=3))
    ax.text(center_x, center_y + 0.055, r"$\hat{u}(x,B)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PURPLE_EDGE, zorder=4)
    ax.text(center_x, center_y - 0.14, "期望效用", ha="center", va="center", fontsize=6.7, color=MUTED, zorder=4)
    return node_anchors(center_x - radius, center_y - radius, radius * 2, radius * 2)


def chip_node(ax, x, y, width, height, text):
    body = Rectangle((x, y), width, height, facecolor=BLUE_FILL, edgecolor=BLUE_EDGE, linewidth=1.1, zorder=3)
    ax.add_patch(body)
    for fraction in (0.25, 0.50, 0.75):
        pin_y = y + height * fraction
        ax.plot([x - 0.10, x], [pin_y, pin_y], color=BLUE_EDGE, linewidth=1.0, zorder=2)
        ax.plot([x + width, x + width + 0.10], [pin_y, pin_y], color=BLUE_EDGE, linewidth=1.0, zorder=2)
    for fraction in (0.28, 0.50, 0.72):
        pin_x = x + width * fraction
        ax.plot([pin_x, pin_x], [y - 0.08, y], color=BLUE_EDGE, linewidth=1.0, zorder=2)
        ax.plot([pin_x, pin_x], [y + height, y + height + 0.08], color=BLUE_EDGE, linewidth=1.0, zorder=2)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=4)
    return node_anchors(x, y, width, height)


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

    # Stage I: each shape communicates a different semantic object.
    a_nodes = [
        parallelogram_node(ax, 0.78, 4.92, 3.98, 0.36, "训练输入", facecolor=WHITE),
        stacked_rollouts(ax, 0.88, 4.18, 3.72, 0.43),
        contrast_node(ax, 0.88, 3.37, 3.72, 0.53),
        hexagon_node(
            ax,
            1.08,
            2.64,
            3.32,
            0.48,
            "经验原子提取",
            facecolor=AMBER_FILL,
            edgecolor=AMBER_EDGE,
            weight="bold",
        ),
        cluster_node(ax, 0.90, 1.89, 3.68, 0.50),
        document_node(ax, 1.02, 1.17, 3.46, 0.46, "记忆卡构造与资格审查"),
        cylinder_node(
            ax,
            0.84,
            0.62,
            3.82,
            0.42,
            "主级记忆库  ·  选择元数据 + 原生前缀 KV",
            fontsize=7.1,
            weight="bold",
        ),
    ]
    for upper, lower in zip(a_nodes, a_nodes[1:]):
        arrow(ax, upper["bottom"], lower["top"])

    # Stage II: retrieval narrows through a funnel; outcomes become utility labels.
    b_nodes = [
        parallelogram_node(ax, 6.18, 4.88, 3.18, 0.38, "验证输入", facecolor=WHITE),
        funnel_node(ax, 6.50, 3.85, 2.54, 0.76, "语义 Top-K\n排除条件感知重排"),
        paired_execution_node(ax, 6.10, 2.92, 3.34, 0.60),
        utility_labels_node(ax, 6.27, 2.03, 3.00, 0.59),
        hexagon_node(
            ax,
            6.31,
            0.84,
            2.92,
            0.66,
            "效用模型\n+ 弃权阈值",
            facecolor=PURPLE_FILL,
            edgecolor=PURPLE_EDGE,
            fontsize=7.5,
            weight="bold",
        ),
    ]
    for upper, lower in zip(b_nodes, b_nodes[1:]):
        arrow(ax, upper["bottom"], lower["top"])

    # Stage III: a frozen selector decides whether the memory cylinder is exposed.
    online_input = parallelogram_node(ax, 10.75, 4.90, 3.70, 0.36, "新输入", facecolor=WHITE)
    online_selector = funnel_node(ax, 11.27, 3.84, 2.66, 0.77, "冻结语义检索器\n+ 适用性重排器")
    expected_utility = score_node(ax, 12.60, 3.17, 0.38)
    arrow(ax, online_input["bottom"], online_selector["top"])
    arrow(ax, online_selector["bottom"], expected_utility["top"])

    diamond_center = (12.60, 2.30)
    diamond_w, diamond_h = 2.24, 0.66
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
    arrow(ax, expected_utility["bottom"], (diamond_center[0], diamond_center[1] + diamond_h / 2))

    prefix = cylinder_node(
        ax,
        10.48,
        1.18,
        2.12,
        0.50,
        "选中的\n原生前缀 KV",
        fontsize=6.9,
        weight="bold",
    )
    no_memory_center = (13.78, 1.43)
    no_memory_patch = Ellipse(
        no_memory_center,
        1.42,
        0.50,
        facecolor=GRAY_FILL,
        edgecolor=LINE,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(no_memory_patch)
    ax.text(*no_memory_center, "不使用记忆", ha="center", va="center", fontsize=7.2, zorder=4)
    no_memory = node_anchors(13.07, 1.18, 1.42, 0.50)
    reasoner = chip_node(ax, 10.72, 0.59, 2.70, 0.43, "冻结推理模型")
    output = parallelogram_node(ax, 13.72, 0.59, 1.04, 0.43, "模型输出", facecolor=WHITE, fontsize=7.1)

    arrow(ax, (diamond_center[0] - 0.58, diamond_center[1] - 0.19), prefix["top"], color=GREEN_EDGE, connectionstyle="arc3,rad=0.06")
    ax.text(11.55, 1.91, "是", ha="center", va="center", fontsize=6.8, color=GREEN_EDGE, fontweight="bold")
    arrow(ax, (diamond_center[0] + 0.58, diamond_center[1] - 0.19), no_memory["top"], color=LINE, connectionstyle="arc3,rad=-0.06")
    ax.text(13.66, 1.91, "否", ha="center", va="center", fontsize=6.8, color=MUTED, fontweight="bold")
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
        [b_nodes[-1]["right"], (10.05, b_nodes[-1]["right"][1]), (10.05, online_selector["left"][1]), online_selector["left"]],
        color=PURPLE_EDGE,
    )
    ax.text(10.05, 2.37, "策略", rotation=90, ha="center", va="center", fontsize=6.3, color=PURPLE_EDGE, fontweight="bold")

    # The selected online KV is loaded from the frozen Primary Bank.
    orthogonal_arrow(
        ax,
        [
            a_nodes[-1]["bottom"],
            (a_nodes[-1]["bottom"][0], 0.24),
            (10.30, 0.24),
            (10.30, prefix["left"][1]),
            prefix["left"],
        ],
        color=GREEN_EDGE,
    )
    ax.text(7.44, 0.15, "冻结的主级记忆库", ha="center", va="center", fontsize=6.6, color=GREEN_EDGE, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = STEM.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    fig.savefig(STEM.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    # Matplotlib emits trailing spaces inside multiline SVG paths. Normalize the
    # generated source so repository whitespace checks remain meaningful.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    draw_figure()
