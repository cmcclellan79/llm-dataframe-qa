"""
Generate publication-quality figures from DataFrame QA experiment results.
Saves all figures as high-res PNGs for poster/paper use.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

TEAL = "#1D9E75"
BLUE = "#378ADD"
PURPLE = "#7F77DD"
AMBER = "#EF9F27"
RED = "#E24B4A"
GRAY = "#888780"


# ========================================================================
# Figure 1: Pass@1 accuracy — metadata-only vs full-table
# ========================================================================

def fig_accuracy():
    questions = [
        "Nationality\nlookup",
        "Count US\nplayers",
        "Michigan\nplayer",
        "Position\nlookup",
        "Unique\nnationalities",
    ]
    meta_scores = [1, 1, 0, 1, 1]
    full_scores = [1, 1, 1, 1, 1]

    x = np.arange(len(questions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, meta_scores, width, label="Metadata-only (80%)", color=TEAL, edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, full_scores, width, label="Full-table (100%)", color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)

    # Highlight the failure
    ax.bar(2 - width/2, 0, width, color=RED, alpha=0.15, zorder=2)
    ax.annotate("Value retrieval\nambiguity", xy=(2 - width/2, 0.05), fontsize=9,
                color=RED, ha="center", style="italic")

    ax.set_ylabel("Correct (1) / Incorrect (0)")
    ax.set_title("Pass@1 accuracy per question: metadata-only vs full-table")
    ax.set_xticks(x)
    ax.set_xticklabels(questions, fontsize=10)
    ax.set_ylim(-0.05, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig("results/fig1_accuracy.png")
    print("  Saved fig1_accuracy.png")
    plt.close()


# ========================================================================
# Figure 2: Token usage comparison (stacked bars)
# ========================================================================

def fig_tokens():
    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    meta_input  = [239, 240, 237, 241, 239]
    meta_output = [29, 26, 36, 31, 17]
    full_input  = [344, 345, 342, 346, 344]
    full_output = [29, 26, 31, 31, 17]

    x = np.arange(len(questions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width/2, meta_input, width, label="Metadata input", color=TEAL, edgecolor="white", linewidth=0.5)
    ax.bar(x - width/2, meta_output, width, bottom=meta_input, label="Metadata output", color="#5DCAA5", edgecolor="white", linewidth=0.5)

    ax.bar(x + width/2, full_input, width, label="Full-table input", color=BLUE, edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, full_output, width, bottom=full_input, label="Full-table output", color="#85B7EB", edgecolor="white", linewidth=0.5)

    # Add savings annotations
    for i in range(len(questions)):
        saving = (full_input[i] + full_output[i]) - (meta_input[i] + meta_output[i])
        mid_y = max(full_input[i] + full_output[i], meta_input[i] + meta_output[i]) + 15
        ax.annotate(f"-{saving}", xy=(x[i], mid_y), fontsize=9, ha="center", color=RED, fontweight="bold")

    ax.set_ylabel("Tokens")
    ax.set_title("Token usage per question: metadata-only vs full-table")
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig("results/fig2_tokens.png")
    print("  Saved fig2_tokens.png")
    plt.close()


# ========================================================================
# Figure 3: Cross-domain training data — complexity breakdown
# ========================================================================

def fig_complexity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Donut chart — complexity distribution
    sizes = [31, 11, 3]
    labels = ["Data analysis\n(69%)", "Retrieval\n(24%)", "Aggregation\n(7%)"]
    colors = [PURPLE, TEAL, AMBER]
    explode = (0.03, 0.03, 0.03)

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda pct: f"{int(round(pct/100.*sum(sizes)))}",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")

    circle = plt.Circle((0, 0), 0.5, fc="white")
    ax1.add_artist(circle)
    ax1.text(0, 0, "45\npairs", ha="center", va="center", fontsize=16, fontweight="bold", color="#333")
    ax1.set_title("Complexity distribution")

    # Grouped bar — pairs by domain and role
    domains = ["Medical", "Automotive", "Sports"]
    roles_data = {
        "Data scientist": [5, 5, 5],
        "General user": [5, 5, 5],
        "Data owner": [5, 5, 5],
    }
    x = np.arange(len(domains))
    width = 0.25
    role_colors = [PURPLE, TEAL, AMBER]

    for i, (role, values) in enumerate(roles_data.items()):
        ax2.bar(x + i * width - width, values, width, label=role,
                color=role_colors[i], edgecolor="white", linewidth=0.5)

    ax2.set_ylabel("Pairs generated")
    ax2.set_title("Pairs by domain and role")
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.set_ylim(0, 7)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Cross-domain training data generated from metadata only", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("results/fig3_crossdomain.png")
    print("  Saved fig3_crossdomain.png")
    plt.close()


# ========================================================================
# Figure 4: Token scaling — metadata stays flat, full-table grows
# ========================================================================

def fig_scaling():
    rows = [7, 50, 100, 500, 1000, 5000, 10000]
    meta_tokens = [240] * len(rows)
    full_tokens = [240 + r * 15 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(rows, full_tokens, "o-", color=RED, label="Full-table (grows with rows)",
                linewidth=2.5, markersize=7, zorder=3)
    ax.semilogy(rows, meta_tokens, "o-", color=TEAL, label="Metadata-only (constant ~240)",
                linewidth=2.5, markersize=7, zorder=3)

    # Context window limits
    ax.axhline(y=4096, color=GRAY, linestyle="--", alpha=0.5, linewidth=1)
    ax.text(rows[-1] * 0.85, 4096 * 1.15, "4K context limit", fontsize=9, color=GRAY, ha="right")

    ax.axhline(y=8192, color=GRAY, linestyle="--", alpha=0.5, linewidth=1)
    ax.text(rows[-1] * 0.85, 8192 * 1.15, "8K context limit", fontsize=9, color=GRAY, ha="right")

    # Fill the savings area
    ax.fill_between(rows, meta_tokens, full_tokens, alpha=0.08, color=RED)

    # Annotate the observed data point
    ax.annotate("Our experiment\n(7 rows, 28% savings)", xy=(7, 371), fontsize=9,
                xytext=(50, 800), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
                color=BLUE, fontweight="bold")

    ax.set_xlabel("Number of rows in dataframe")
    ax.set_ylabel("Input tokens per query (log scale)")
    ax.set_title("Token scaling: metadata-only remains constant regardless of table size")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig("results/fig4_scaling.png")
    print("  Saved fig4_scaling.png")
    plt.close()


# ========================================================================
# Figure 5: Pipeline overview — the retrieval-based approach
# ========================================================================

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5,  1.2, 2.2, 1.6, "Schema\nMetadata", BLUE, "Column names\n+ data types"),
        (3.2,  1.2, 2.2, 1.6, "Generate\nTraining Pairs", PURPLE, "Claude API\n(metadata only)"),
        (5.9,  1.2, 2.2, 1.6, "TF-IDF\nRetrieval Index", TEAL, "Vectorize\nquestions"),
        (8.6,  1.2, 2.2, 1.6, "Retrieve &\nAdapt Query", AMBER, "Lightweight\nAPI call"),
        (11.3, 1.2, 2.2, 1.6, "Execute\nLocally", RED, "Sandboxed\nPandas"),
    ]

    for x, y, w, h, title, color, subtitle in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h * 0.65, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x + w/2, y + h * 0.25, subtitle, ha="center", va="center",
                fontsize=8.5, color=GRAY)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

    # Labels
    ax.text(1.6, 3.2, "ONE-TIME SETUP", fontsize=9, color=BLUE, fontweight="bold", ha="center")
    ax.annotate("", xy=(5.0, 3.0), xytext=(1.6, 3.0),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1, linestyle="--"))

    ax.text(10.95, 3.2, "PER-QUESTION", fontsize=9, color=AMBER, fontweight="bold", ha="center")
    ax.annotate("", xy=(12.4, 3.0), xytext=(9.5, 3.0),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1, linestyle="--"))

    ax.text(7.0, 0.5, "Data never leaves your machine", fontsize=10, ha="center",
            color=TEAL, fontweight="bold", style="italic")

    ax.set_title("Retrieval-based DataFrame QA pipeline", fontsize=14, fontweight="bold", pad=20)

    fig.savefig("results/fig5_pipeline.png")
    print("  Saved fig5_pipeline.png")
    plt.close()


# ========================================================================
# Figure 6: Summary stats comparison card
# ========================================================================

def fig_summary():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    # Three comparison columns
    cols = [
        ("Metadata-Only\n(This Work)", TEAL, [
            ("Input tokens/query", "~240"),
            ("Data exposure", "None"),
            ("pass@1 (7 rows)", "80%"),
            ("Scales to", "Any size"),
        ]),
        ("Full-Table\n(Traditional)", RED, [
            ("Input tokens/query", "~345+"),
            ("Data exposure", "All rows"),
            ("pass@1 (7 rows)", "100%"),
            ("Scales to", "Context limit"),
        ]),
        ("Retrieval-Based\n(Extension)", PURPLE, [
            ("Input tokens/query", "~200*"),
            ("Data exposure", "None"),
            ("Training pairs", "45 from metadata"),
            ("Domains", "3 (cross-domain)"),
        ]),
    ]

    for i, (title, color, metrics) in enumerate(cols):
        x_start = 0.5 + i * 4.5
        # Header
        rect = mpatches.FancyBboxPatch((x_start, 2.2), 3.8, 1.0, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_start + 1.9, 2.7, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

        # Metrics
        for j, (label, value) in enumerate(metrics):
            y = 1.6 - j * 0.5
            ax.text(x_start + 0.2, y, label, fontsize=9, color=GRAY, va="center")
            ax.text(x_start + 3.6, y, value, fontsize=10, fontweight="bold",
                    color=color, va="center", ha="right")

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 3.5)
    ax.text(7, -0.3, "* Estimated: retrieval reduces prompt size by reusing cached templates",
            fontsize=8, color=GRAY, ha="center", style="italic")

    fig.savefig("results/fig6_summary.png")
    print("  Saved fig6_summary.png")
    plt.close()


# ========================================================================

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    print("\nGenerating figures...\n")
    fig_accuracy()
    fig_tokens()
    fig_complexity()
    fig_scaling()
    fig_pipeline()
    fig_summary()
    print(f"\nAll figures saved to results/")
