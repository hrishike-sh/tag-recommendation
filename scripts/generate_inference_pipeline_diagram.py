import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_inference_diagram():
    # Academic clean typography
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # Wide canvas for systems architecture diagram (17.2 x 9.4 inches)
    fig, ax = plt.subplots(figsize=(17.2, 9.4), dpi=300)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 9.4)
    ax.axis('off')

    # Color palette
    c_bg_user = '#F8FAFC'      # Slate 50
    c_bg_catalog = '#EFF6FF'   # Soft blue
    c_bg_scores = '#F0FDF4'    # Soft green
    c_bg_fusion = '#FFF1F2'    # Soft rose
    c_bg_filter = '#F5F3FF'    # Soft purple
    c_bg_topk = '#ECFDF5'      # Soft emerald
    c_callout = '#FEF3C7'      # Amber tint for callouts

    border_slate = '#475569'
    border_blue = '#2563EB'
    border_green = '#059669'
    border_red = '#DC2626'
    border_purple = '#7C3AED'
    border_emerald = '#047857'
    border_amber = '#D97706'

    text_dark = '#0F172A'
    text_muted = '#475569'

    # Helper: draw a styled box
    def draw_box(x, y, w, h, bg_color, title="", subtitle="", items=None, border_color=border_slate, lw=1.2, radius=0.12):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=border_color,
            facecolor=bg_color,
            zorder=2
        )
        ax.add_patch(box)

        cur_y = y + h - 0.28
        if title:
            ax.text(x + w/2, cur_y, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=text_dark, zorder=3)
            cur_y -= 0.22
        
        if subtitle:
            ax.text(x + w/2, cur_y, subtitle, ha='center', va='center', fontsize=7.6, fontstyle='italic', color=text_muted, zorder=3)
            cur_y -= 0.26
        else:
            cur_y -= 0.04

        if items:
            for it in items:
                if isinstance(it, tuple):
                    text, is_bold, is_italic, col, font_size = it
                    fw = 'bold' if is_bold else 'normal'
                    fs = 'italic' if is_italic else 'normal'
                    ax.text(x + w/2, cur_y, text, ha='center', va='center', fontsize=font_size, fontweight=fw, fontstyle=fs, color=col, zorder=3)
                elif it == "":
                    cur_y -= 0.06
                    continue
                else:
                    ax.text(x + w/2, cur_y, it, ha='center', va='center', fontsize=7.6, color=text_dark, zorder=3)
                cur_y -= 0.24

    # Helper: draw arrow with optional badge
    def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, color='#475569', label_offset=(0, 0.12), badge_bg='white', badge_edge='#CBD5E1', lw=1.3):
        ax.annotate(
            "",
            xy=(x2, y2), xycoords='data',
            xytext=(x1, y1), textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=12,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2)/2 + label_offset[0], (y1 + y2)/2 + label_offset[1]
            ax.text(mx, my, label, ha='center', va='center', fontsize=7.5, color=text_dark, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2,rounding_size=0.08', facecolor=badge_bg, edgecolor=badge_edge, lw=0.8, alpha=0.95))

    # Helper: draw section header badge
    def draw_stage_badge(x, y, w, h, text, bg_color, border_color):
        badge = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            linewidth=1.0,
            edgecolor=border_color,
            facecolor=bg_color,
            zorder=3
        )
        ax.add_patch(badge)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.0, fontweight='bold', color=border_color, zorder=4)

    # -------------------------------------------------------------
    # 1. Main Header Banner
    # -------------------------------------------------------------
    ax.text(0.4, 9.05, "FULL-CATALOG INFERENCE & EXACT TOP-K RECOMMENDATION PIPELINE", 
            ha='left', va='center', fontsize=12.0, fontweight='bold', color=text_dark)
    ax.text(0.4, 8.78, r"Per-user on-the-fly vectorized scoring of 1,493,688 candidate tracks, per-user score calibration, and exact Top-K ranking without approximate indexing", 
            ha='left', va='center', fontsize=8.5, fontstyle='italic', color=text_muted)
    ax.plot([0.4, 16.8], [8.60, 8.60], color='#CBD5E1', lw=1.1)

    # -------------------------------------------------------------
    # 2. Stage 1: User Query & Latent Representations (x: 0.4 to 2.8)
    # -------------------------------------------------------------
    draw_stage_badge(0.4, 8.22, 2.5, 0.26, "STAGE 1: USER QUERY VECTORS", '#F1F5F9', border_slate)

    draw_box(
        x=0.4, y=4.5, w=2.5, h=3.6,
        bg_color=c_bg_user,
        border_color=border_slate,
        lw=1.3,
        title=r"TARGET USER: $u \in \mathcal{U}$",
        subtitle="Evaluated User Representations",
        items=[
            ("Learned Collaborative Factor:", True, False, border_blue, 7.8),
            (r"$x_u \in \mathbb{R}^{64}$", True, False, '#1E40AF', 8.6),
            ("Implicit MSVD ALS factors", False, True, text_muted, 7.3),
            ("", False, False, text_dark, 7.0),
            ("Semantic Preference Profile:", True, False, border_green, 7.8),
            (r"$z_u = \frac{C_u T}{\|C_u T\|_2} \in \mathbb{R}^{2{,}299}$", True, False, '#065F46', 8.2),
            ("Confidence-projected HetRec tags", False, True, text_muted, 7.3),
            ("", False, False, text_dark, 7.0),
            ("Temporal Recency Vector:", True, False, border_amber, 7.8),
            (r"$S_u^{\mathrm{temp}} \in \mathbb{R}^{1{,}493{,}688}$", True, False, '#92400E', 8.2),
            ("Exponentially decayed history", False, True, text_muted, 7.3)
        ]
    )

    # Callout 1: Memory & Matrix Allocation Invariant
    draw_box(
        x=0.4, y=0.5, w=2.5, h=3.7,
        bg_color=c_callout,
        border_color=border_amber,
        lw=1.2,
        title="SYSTEM INVARIANT",
        subtitle="Memory Efficiency Guarantees",
        items=[
            ("Strict Memory Constraint:", True, False, border_amber, 7.8),
            (r"$\mathbf{No \; dense \; 992 \times 1{,}493{,}688}$", True, False, '#78350F', 8.0),
            (r"$\mathbf{score \; matrix \; is \; materialized.}$", True, False, '#78350F', 8.0),
            ("", False, False, text_dark, 7.0),
            ("Representation:", True, False, text_dark, 7.6),
            ("Sparse CSR representations;", False, False, text_dark, 7.5),
            ("no dense U x I matrix", False, False, text_dark, 7.5),
            ("allocation during inference.", False, False, text_dark, 7.5),
            ("", False, False, text_dark, 7.0),
            ("Per-user evaluation vector", False, True, text_muted, 7.3),
            ("allocated dynamically in RAM", False, True, text_muted, 7.3)
        ]
    )

    # -------------------------------------------------------------
    # 3. Stage 2: 1.49M Candidate Catalog Scoring (x: 3.3 to 6.3)
    # -------------------------------------------------------------
    draw_stage_badge(3.3, 8.22, 3.1, 0.26, r"STAGE 2: FULL CATALOG SCORING ($|\mathcal{I}| = 1{,}493{,}688$)", '#DBEAFE', border_blue)

    # Visual depiction of candidate catalog stack
    catalog_box = patches.FancyBboxPatch(
        (3.3, 4.5), 3.1, 3.6,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=border_blue,
        facecolor=c_bg_catalog,
        zorder=2
    )
    ax.add_patch(catalog_box)

    ax.text(4.85, 7.82, "EVALUATED CATALOG", ha='center', va='center', fontsize=9.0, fontweight='bold', color=text_dark, zorder=3)
    ax.text(4.85, 7.58, r"All $1{,}493{,}688$ Candidate Items", ha='center', va='center', fontsize=7.6, fontstyle='italic', color=text_muted, zorder=3)

    # Visual candidate item slice grid/stack
    slice_colors = ['#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6']
    for idx, sy in enumerate([7.15, 6.75, 6.35, 5.95]):
        sbox = patches.Rectangle((3.55, sy), 2.6, 0.28, facecolor=slice_colors[idx], edgecolor='#1D4ED8', lw=0.8, zorder=3)
        ax.add_patch(sbox)
        if idx == 0:
            ax.text(4.85, sy + 0.14, r"Item Factor Matrix: $Y \in \mathbb{R}^{1{,}493{,}688 \times 64}$", ha='center', va='center', fontsize=7.3, color='#1E3A8A', fontweight='bold', zorder=4)
        elif idx == 1:
            ax.text(4.85, sy + 0.14, r"Tag Matrix (Sparse CSR): $T \in \mathbb{R}^{1{,}493{,}688 \times 2{,}299}$", ha='center', va='center', fontsize=7.3, color='#1E3A8A', fontweight='bold', zorder=4)
        elif idx == 2:
            ax.text(4.85, sy + 0.14, r"Temporal Sparse Vector: $S_u^{\mathrm{temp}} \in \mathbb{R}^{1{,}493{,}688}$", ha='center', va='center', fontsize=7.3, color='#1E3A8A', fontweight='bold', zorder=4)
        elif idx == 3:
            ax.text(4.85, sy + 0.14, r"$\dots \; 1{,}493{,}688 \text{ Tracks Pre-Indexed} \; \dots$", ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#1E3A8A', zorder=4)

    # Description under the catalog stack
    ax.text(4.85, 5.50, "Full-Catalog Vectorized Computations:", ha='center', va='center', fontsize=7.6, fontweight='bold', color=text_dark, zorder=3)
    ax.text(4.85, 5.24, r"MSVD: $s^{\mathrm{msvd}} = Y x_u \quad (\text{BLAS GEMV, length } 1.49\text{M})$", ha='center', va='center', fontsize=7.2, color='#1E40AF', zorder=3)
    ax.text(4.85, 4.98, r"Tag: $s^{\mathrm{tag}} = T z_u \quad (\text{Sparse CSR SpMV, length } 1.49\text{M})$", ha='center', va='center', fontsize=7.2, color='#065F46', zorder=3)
    ax.text(4.85, 4.72, r"Temporal: $s^{\mathrm{temporal}} = S_u^{\mathrm{temp}} \quad (\text{Sparse Lookup})$", ha='center', va='center', fontsize=7.2, color='#92400E', zorder=3)

    # Connecting arrows from User to Catalog
    draw_arrow(2.9, 6.3, 3.3, 6.3, label=r"$x_u, z_u, S_u$", color=border_blue, label_offset=(0, 0.14))

    # Stage 2 Bottom Box: Three Score Vectors
    draw_box(
        x=3.3, y=0.5, w=3.1, h=3.7,
        bg_color=c_bg_scores,
        border_color=border_green,
        lw=1.2,
        title="THREE RAW SCORE VECTORS",
        subtitle=r"Length: $1{,}493{,}688$ per user",
        items=[
            ("1. Collaborative Dot-Products:", True, False, border_blue, 7.6),
            (r"$s^{\mathrm{msvd}} \in (-\infty, \infty)^{1{,}493{,}688}$", True, False, '#1E40AF', 8.0),
            ("Unbounded cross-user variance", False, True, text_muted, 7.2),
            ("", False, False, text_dark, 7.0),
            ("2. Semantic Cosine Similarities:", True, False, border_green, 7.6),
            (r"$s^{\mathrm{tag}} \in [0, 1]^{1{,}493{,}688}$", True, False, '#065F46', 8.0),
            ("Sparse tag overlap (39.78% catalog)", False, True, text_muted, 7.2),
            ("", False, False, text_dark, 7.0),
            ("3. Temporal Recency Scores:", True, False, border_amber, 7.6),
            (r"$s^{\mathrm{temporal}} \in [0, \infty)^{1{,}493{,}688}$", True, False, '#92400E', 8.0),
            ("Exponentially decayed interaction weights", False, True, text_muted, 7.2)
        ]
    )

    # Vertical arrow from Catalog stack to Raw Score Vectors
    draw_arrow(4.85, 4.5, 4.85, 4.2, label="Generate 3 Vectors", color=border_blue, label_offset=(0.85, 0.0))

    # -------------------------------------------------------------
    # 4. Stage 3: Normalization & Multi-Stream Fusion (x: 6.8 to 10.1)
    # -------------------------------------------------------------
    draw_stage_badge(6.8, 8.22, 3.3, 0.26, "STAGE 3: NORMALIZATION & FUSION", '#FCE7F3', border_red)

    # Box 3A: Per-User Normalization
    draw_box(
        x=6.8, y=4.5, w=3.3, h=3.6,
        bg_color=c_bg_fusion,
        border_color=border_red,
        lw=1.3,
        title="PER-USER SCORE NORMALIZATION",
        subtitle="Aligning Scales Across Streams",
        items=[
            ("Collaborative Min-Max Calibration:", True, False, border_blue, 7.8),
            (r"$\tilde{s}_i^{\mathrm{msvd}} = \frac{s_i^{\mathrm{msvd}} - \min_j s_j^{\mathrm{msvd}}}{\max_j s_j^{\mathrm{msvd}} - \min_j s_j^{\mathrm{msvd}}} \in [0, 1]$", True, False, '#1E40AF', 7.8),
            ("Evaluated across all 1,493,688 candidates", False, True, text_muted, 7.2),
            ("", False, False, text_dark, 7.0),
            ("Semantic Cosine Scaling:", True, False, border_green, 7.8),
            (r"$\tilde{s}_i^{\mathrm{tag}} = s_i^{\mathrm{tag}} \in [0, 1]$", True, False, '#065F46', 8.0),
            ("", False, False, text_dark, 7.0),
            ("Temporal L2 Unit Scaling:", True, False, border_amber, 7.8),
            (r"$\tilde{s}_u^{\mathrm{temporal}} = \frac{s_u^{\mathrm{temporal}}}{\|s_u^{\mathrm{temporal}}\|_2} \in [0, 1]$", True, False, '#92400E', 7.8)
        ]
    )

    # Arrow from Raw Scores to Normalization
    draw_arrow(6.4, 2.35, 6.8, 5.2, label=r"Raw $s$", color=border_green, rad=0.08, label_offset=(0, 0.16))

    # Box 3B: Weighted Linear Fusion
    draw_box(
        x=6.8, y=0.5, w=3.3, h=3.7,
        bg_color=c_bg_fusion,
        border_color=border_red,
        lw=1.4,
        title="WEIGHTED MULTI-STREAM FUSION",
        subtitle="Global Convex Combination",
        items=[
            ("Validation-Optimized Scoring:", True, False, border_red, 7.8),
            (r"$\hat{r}_{ui} = 0.20 \, \tilde{s}_{ui}^{\mathrm{msvd}} + 0.30 \, \tilde{s}_{ui}^{\mathrm{tag}} + 0.50 \, \tilde{s}_{ui}^{\mathrm{temporal}}$", True, False, '#991B1B', 8.0),
            ("", False, False, text_dark, 7.0),
            ("Composite Score Properties:", True, False, text_dark, 7.6),
            ("Length: 1,493,688 full candidate vector", False, False, text_dark, 7.4),
            ("Convex sum of weights: 0.20 + 0.30 + 0.50 = 1.00", False, True, text_muted, 7.3),
            ("Fixed globally across all 992 users", False, True, text_muted, 7.3),
            ("", False, False, text_dark, 7.0),
            ("No score thresholding applied", True, False, '#7C3AED', 7.4),
            ("Every candidate track retains a valid score", False, True, text_muted, 7.2)
        ]
    )

    # Arrow from Normalization to Fusion
    draw_arrow(8.45, 4.5, 8.45, 4.2, label=r"$\tilde{s}^{\mathrm{msvd}}, \tilde{s}^{\mathrm{tag}}, \tilde{s}^{\mathrm{temp}}$", color=border_red, label_offset=(0.95, 0.0))

    # -------------------------------------------------------------
    # 5. Stage 4: Filtering & Exact Top-K Selection (x: 10.6 to 13.9)
    # -------------------------------------------------------------
    draw_stage_badge(10.6, 8.22, 3.3, 0.26, "STAGE 4: FILTERING & EXACT PARTIAL SORT", '#EDE9FE', border_purple)

    # Box 4A: Historical Item Filtering
    draw_box(
        x=10.6, y=4.5, w=3.3, h=3.6,
        bg_color=c_bg_filter,
        border_color=border_purple,
        lw=1.3,
        title="HISTORICAL-ITEM FILTERING",
        subtitle=r"Active Candidate Space: $\mathcal{I} \setminus \mathcal{H}_u$",
        items=[
            ("Historical Interaction Set:", True, False, text_dark, 7.8),
            (r"$\mathcal{H}_u = \mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}}$", True, False, '#6D28D9', 8.2),
            ("All previously consumed tracks", False, True, text_muted, 7.2),
            ("", False, False, text_dark, 7.0),
            ("In-Place Score Masking:", True, False, border_purple, 7.8),
            (r"$\hat{r}_{ui} \leftarrow -\infty \quad \forall i \in \mathcal{H}_u$", True, False, '#5B21B6', 8.6),
            ("Guarantees repeat listening events", False, True, text_muted, 7.2),
            ("cannot enter recommendation rank list", False, True, text_muted, 7.2),
            ("", False, False, text_dark, 7.0),
            ("Zero Training Leakage Enforced", True, False, border_purple, 7.6)
        ]
    )

    # Arrow from Fusion to Filtering
    draw_arrow(10.1, 2.35, 10.6, 5.2, label=r"$\hat{r}_u \in \mathbb{R}^{1.49\mathrm{M}}$", color=border_red, rad=0.08, label_offset=(0, 0.16))

    # Box 4B: Exact Top-K Selection via numpy.argpartition
    draw_box(
        x=10.6, y=0.5, w=3.3, h=3.7,
        bg_color=c_bg_topk,
        border_color=border_emerald,
        lw=1.5,
        title="EXACT TOP-K SELECTION",
        subtitle=r"$\operatorname{arg\,max}$ over 1,493,688 Candidates",
        items=[
            ("Exact Partial Sort Algorithm:", True, False, border_emerald, 7.8),
            (r"$\mathbf{numpy.argpartition}(-\hat{r}_u, K)$", True, False, '#047857', 8.6),
            (r"Sub-linear $O(|\mathcal{I}|)$ worst-case complexity", False, True, text_muted, 7.3),
            ("", False, False, text_dark, 7.0),
            ("Full Catalog Consideration:", True, False, text_dark, 7.6),
            ("All 1,493,688 candidate scores", False, False, text_dark, 7.4),
            ("are partitioned; no pruning", False, False, text_dark, 7.4),
            ("or pre-filtering of unseen items.", False, False, text_dark, 7.4),
            ("", False, False, text_dark, 7.0),
            ("Measured End-to-End Latency:", True, False, border_emerald, 7.6),
            ("96.48 ms per user across full catalog", False, True, text_muted, 7.3)
        ]
    )

    # Arrow from Filtering to Top-K Selection
    draw_arrow(12.25, 4.5, 12.25, 4.2, label="Masked Scores", color=border_purple, label_offset=(0.75, 0.0))

    # -------------------------------------------------------------
    # 6. Stage 5: Final Output & Callout Annotations (x: 14.3 to 16.8)
    # -------------------------------------------------------------
    draw_stage_badge(14.3, 8.22, 2.5, 0.26, "STAGE 5: TOP-10 OUTPUT", '#D1FAE5', border_emerald)

    # Ranked Top-10 Output List
    draw_box(
        x=14.3, y=3.7, w=2.5, h=4.4,
        bg_color=c_bg_topk,
        border_color=border_emerald,
        lw=1.4,
        title="TOP-10 RECOMMENDATIONS",
        subtitle="Final Recommended Tracks",
        items=[
            (r"$\mathbf{Rank \; 1:} \quad \text{Track } i_1^* \quad (\hat{r} = 0.842)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 2:} \quad \text{Track } i_2^* \quad (\hat{r} = 0.819)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 3:} \quad \text{Track } i_3^* \quad (\hat{r} = 0.791)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 4:} \quad \text{Track } i_4^* \quad (\hat{r} = 0.768)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 5:} \quad \text{Track } i_5^* \quad (\hat{r} = 0.745)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 6:} \quad \text{Track } i_6^* \quad (\hat{r} = 0.730)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 7:} \quad \text{Track } i_7^* \quad (\hat{r} = 0.712)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 8:} \quad \text{Track } i_8^* \quad (\hat{r} = 0.699)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 9:} \quad \text{Track } i_9^* \quad (\hat{r} = 0.684)$", False, False, '#064E3B', 7.4),
            (r"$\mathbf{Rank \; 10:} \quad \text{Track } i_{10}^* \quad (\hat{r} = 0.671)$", False, False, '#064E3B', 7.4),
            ("", False, False, text_dark, 7.0),
            ("Sorted Top-K: Exact Highest Scores", True, False, border_emerald, 7.4)
        ]
    )

    # Arrow from Top-K Selection to Ranked List
    draw_arrow(13.9, 2.35, 14.3, 4.8, label="Top 10", color=border_emerald, rad=0.08, label_offset=(0, 0.16))

    # Callout 2: Exact Retrieval Guarantee
    draw_box(
        x=14.3, y=0.5, w=2.5, h=3.0,
        bg_color=c_callout,
        border_color=border_amber,
        lw=1.2,
        title="EXACTNESS GUARANTEE",
        subtitle="Algorithmic Integrity",
        items=[
            ("Key Retrieval Property:", True, False, border_amber, 7.8),
            (r"$\mathbf{Exact \; retrieval; \; no}$", True, False, '#78350F', 8.2),
            (r"$\mathbf{approximate \; nearest\text{-}}$", True, False, '#78350F', 8.2),
            (r"$\mathbf{neighbor \; index}$", True, False, '#78350F', 8.2),
            ("", False, False, text_dark, 7.0),
            ("100% recall of top scores", True, False, text_dark, 7.5),
            ("relative to mathematical", False, False, text_dark, 7.5),
            ("objective across catalog.", False, False, text_dark, 7.5)
        ]
    )

    plt.tight_layout()

    # Save outputs
    pdf_path = "paper/inference_topk_pipeline.pdf"
    svg_path = "paper/inference_topk_pipeline.svg"
    png_path = "paper/inference_topk_pipeline.png"

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Successfully generated inference pipeline diagram to:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == '__main__':
    generate_inference_diagram()
