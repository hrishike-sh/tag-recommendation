import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    # Academic publication typography
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    paper_dir = Path("paper")
    phase9_fig_dir = Path("results/phase9/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    phase9_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load authoritative data
    table4_path = Path("results/phase9/paper_tables/table4_test_ablation.json")
    table5_path = Path("results/phase9/paper_tables/table5_activity_segments.json")
    bootstrap_path = Path("results/phase9/statistics/bootstrap.json")

    with open(table4_path, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)
    with open(table5_path, "r", encoding="utf-8") as f:
        activity_data = json.load(f)
    with open(bootstrap_path, "r", encoding="utf-8") as f:
        bootstrap_data = json.load(f)

    # =========================================================================
    # 1. ACTIVITY PLOT: NDCG@10 by User Activity Segment
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

    tiers = [
        "Low Activity\n(N=172, ≤ 2,090 plays)",
        "Mid Activity\n(N=172, 2,092–5,018)",
        "High Activity\n(N=174, ≥ 5,028 plays)"
    ]
    tier_keys = ["low_activity", "mid_activity", "high_activity"]

    base_vals = [activity_data[k]["models"]["Base_MSVD"]["NDCG@10"] for k in tier_keys]
    arb_vals = [activity_data[k]["models"]["Global_Arbiter"]["NDCG@10"] for k in tier_keys]

    x = np.arange(len(tiers))
    width = 0.32

    # Restrained neutral academic colors: Slate grey vs Academic Navy
    c_base = "#64748B"    # Slate 500
    c_arb = "#2563EB"     # Royal Blue 600

    rects1 = ax.bar(x - width/2, base_vals, width, label="Base MSVD", color=c_base, edgecolor="#334155", lw=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, arb_vals, width, label="Global Normalized Arbiter", color=c_arb, edgecolor="#1E40AF", lw=0.9, zorder=3)

    # Value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.0004, f"{h:.6f}", ha='center', va='bottom', fontsize=7.5, color="#1E293B")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.0004, f"{h:.6f}", ha='center', va='bottom', fontsize=7.5, color="#1E293B", fontweight='bold' if h > 0.01 else 'normal')

    # Annotation highlighting the concentration of gain in low-activity cohort without claiming overall win
    ax.annotate(
        "+50.47% Relative Difference\n(Complementary sparse signals)",
        xy=(0 + width/2, arb_vals[0]), xytext=(0.35, 0.0185),
        arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.0, shrinkB=4),
        fontsize=7.8, fontweight='bold', color="#1E40AF",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.8)
    )

    ax.annotate(
        "Base MSVD higher in dense cohorts\n(Global weights do not benefit dense users)",
        xy=(2 - width/2, base_vals[2] + 0.0008), xytext=(1.1, 0.0088),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0, shrinkB=4),
        fontsize=7.5, color="#334155", fontstyle='italic',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8)
    )

    ax.set_ylabel("NDCG@10", fontsize=9.5, fontweight='bold', color="#0F172A")
    ax.set_title("NDCG@10 by User Activity Segment", fontsize=11.0, fontweight='bold', color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=8.5, color="#0F172A")
    ax.set_ylim(0, 0.0225)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for d in [paper_dir, phase9_fig_dir]:
        fig.savefig(d / "activity_tier_breakdown.png", dpi=300)
        fig.savefig(d / "activity_tier_breakdown.pdf")
        fig.savefig(d / "activity_tier_breakdown.svg")
    plt.close()

    # =========================================================================
    # 2. BOOTSTRAP PLOT: Bootstrap Confidence Intervals for Paired Performance Differences
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=300)

    metrics_ci = ["NDCG@10", "NDCG@20", "MAP@10"]
    y_pos = np.arange(len(metrics_ci))

    diff_means = [bootstrap_data[m]["Difference"]["mean"] for m in metrics_ci]
    ci_lows = [bootstrap_data[m]["Difference"]["ci_95_lower"] for m in metrics_ci]
    ci_highs = [bootstrap_data[m]["Difference"]["ci_95_upper"] for m in metrics_ci]

    xerr = [
        [diff_means[i] - ci_lows[i] for i in range(3)],
        [ci_highs[i] - diff_means[i] for i in range(3)]
    ]

    # Neutral styling: slate point estimate markers and crisp error bars (no red/green signaling)
    point_color = "#0F172A"   # Dark slate
    bar_color = "#334155"     # Slate 700

    ax.errorbar(
        diff_means, y_pos, xerr=xerr,
        fmt="o", color=point_color, ecolor=bar_color,
        elinewidth=1.6, capsize=5, capthick=1.4, markersize=6.5, zorder=4
    )

    # Vertical reference line at zero
    ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(0.0001, -0.38, "Zero Difference (No Effect)", ha='left', va='center', fontsize=7.8, color="#64748B", fontstyle='italic')

    # Annotate exact intervals explicitly showing crossing of zero
    for i, m in enumerate(metrics_ci):
        mean_v = diff_means[i]
        low_v = ci_lows[i]
        high_v = ci_highs[i]
        ci_str = f"Mean: {mean_v:+.4f}\n95% CI: [{low_v:+.4f}, {high_v:+.4f}] (Spans Zero)"
        ax.text(high_v + 0.0003, y_pos[i], ci_str, va='center', ha='left', fontsize=7.6, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.6))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_ci, fontsize=9.0, fontweight='bold', color="#0F172A")
    ax.set_xlabel("Mean Paired Difference (Global Normalized Arbiter − Base MSVD)", fontsize=9.0, fontweight='bold', color="#0F172A")
    ax.set_title("Bootstrap Confidence Intervals for Paired Performance Differences", fontsize=10.5, fontweight='bold', color="#0F172A", pad=12)
    ax.set_xlim(-0.0045, 0.0090)
    ax.set_ylim(-0.55, 2.45)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=1)

    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for d in [paper_dir, phase9_fig_dir]:
        fig.savefig(d / "bootstrap_confidence_intervals.png", dpi=300)
        fig.savefig(d / "bootstrap_confidence_intervals.pdf")
        fig.savefig(d / "bootstrap_confidence_intervals.svg")
    plt.close()

    # =========================================================================
    # 3. ABLATION PLOT: NDCG@10 and NDCG@20 Across Model Variants
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)

    # Requested abbreviated x-axis labels
    models_short = [
        "Popularity",
        "Base MSVD",
        "Tag-Fused",
        "Temporal",
        "Fixed Fusion",
        "Global Arbiter",
        "Adaptive Arbiter"
    ]

    # Map model rows in authoritative table4 order
    model_name_map = {
        "Popularity": "Popularity",
        "Base Implicit MSVD": "Base MSVD",
        "Tag-Fused MSVD": "Tag-Fused",
        "Temporal MSVD": "Temporal",
        "Fixed Tag+Temporal": "Fixed Fusion",
        "Global Normalized Arbiter": "Global Arbiter",
        "Adaptive User Arbiter": "Adaptive Arbiter"
    }

    row_dict = {model_name_map[row["Model"]]: row for row in ablation_data if row["Model"] in model_name_map}

    ndcg10_vals = [row_dict[m]["NDCG@10"] for m in models_short]
    ndcg20_vals = [row_dict[m]["NDCG@20"] for m in models_short]

    x = np.arange(len(models_short))
    width = 0.35

    c_ndcg10 = "#1E40AF"   # Deep Navy / Blue
    c_ndcg20 = "#059669"   # Forest Green

    rects1 = ax.bar(x - width/2, ndcg10_vals, width, label="NDCG@10", color=c_ndcg10, edgecolor="#172554", lw=0.8, zorder=3)
    rects2 = ax.bar(x + width/2, ndcg20_vals, width, label="NDCG@20", color=c_ndcg20, edgecolor="#064E3B", lw=0.8, zorder=3)

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.00015, f"{h:.4f}", ha='center', va='bottom', fontsize=6.8, color="#0F172A", rotation=45)
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.00015, f"{h:.4f}", ha='center', va='bottom', fontsize=6.8, color="#0F172A", rotation=45)

    ax.set_ylabel("NDCG Score", fontsize=9.2, fontweight='bold', color="#0F172A")
    ax.set_title("NDCG@10 and NDCG@20 Across Model Variants", fontsize=10.5, fontweight='bold', color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_short, fontsize=8.2, color="#0F172A")
    ax.set_ylim(0, 0.0098)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.2, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for d in [paper_dir, phase9_fig_dir]:
        fig.savefig(d / "ndcg_comparison.png", dpi=300)
        fig.savefig(d / "ndcg_comparison.pdf")
        fig.savefig(d / "ndcg_comparison.svg")
    plt.close()

    print("Successfully generated all three refined plots (PNG, PDF, SVG):")
    print(" - activity_tier_breakdown")
    print(" - bootstrap_confidence_intervals")
    print(" - ndcg_comparison")

if __name__ == "__main__":
    generate_plots()
