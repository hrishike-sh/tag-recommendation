import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_scoring_pipeline_diagram():
    # Academic clean typography
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # Wide canvas for publication-grade layout (17.2 x 9.2 inches)
    fig, ax = plt.subplots(figsize=(17.2, 9.2), dpi=300)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # Color palette - clean academic tones with high contrast
    c_query = '#F8FAFC'       # Slate 50
    c_stream1 = '#EFF6FF'     # Soft blue (Collaborative)
    c_stream2 = '#F0FDF4'     # Soft green (Semantic)
    c_stream3 = '#FFFBEB'     # Soft amber (Temporal)
    c_fusion = '#FFF1F2'      # Soft rose (Arbiter)
    c_filter = '#F5F3FF'      # Soft purple (Filtering)
    c_out = '#ECFDF5'         # Soft mint (Top-K)

    border_slate = '#475569'
    border_blue = '#2563EB'
    border_green = '#059669'
    border_amber = '#D97706'
    border_red = '#DC2626'
    border_purple = '#7C3AED'
    border_mint = '#047857'

    text_dark = '#0F172A'
    text_muted = '#475569'

    # Helper: draw a styled box
    def draw_box(x, y, w, h, bg_color, title, subtitle="", items=None, border_color=border_slate, lw=1.2, radius=0.14):
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
            ax.text(x + w/2, cur_y, title, ha='center', va='center', fontsize=9.2, fontweight='bold', color=text_dark, zorder=3)
            cur_y -= 0.22
        
        if subtitle:
            ax.text(x + w/2, cur_y, subtitle, ha='center', va='center', fontsize=7.8, fontstyle='italic', color=text_muted, zorder=3)
            cur_y -= 0.26
        else:
            cur_y -= 0.06

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
                    ax.text(x + w/2, cur_y, it, ha='center', va='center', fontsize=7.8, color=text_dark, zorder=3)
                cur_y -= 0.25

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
            ax.text(mx, my, label, ha='center', va='center', fontsize=7.6, fontweight='bold' if 'w_' in label else 'normal',
                    color=color if 'w_' in label else text_dark, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2,rounding_size=0.08', facecolor=badge_bg, edgecolor=badge_edge, lw=0.8, alpha=0.95))

    # Helper: draw stream track title badge
    def draw_stream_badge(x, y, w, h, text, bg_color, border_color):
        badge = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=border_color,
            facecolor=bg_color,
            zorder=3
        )
        ax.add_patch(badge)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.2, fontweight='bold', color=border_color, zorder=4)

    # -------------------------------------------------------------
    # 1. Top Header Banner
    # -------------------------------------------------------------
    ax.text(0.4, 8.85, "MATHEMATICAL SCORING PIPELINE OF THE GLOBAL NORMALIZED MULTI-STREAM ARBITER", 
            ha='left', va='center', fontsize=12.0, fontweight='bold', color=text_dark)
    ax.text(0.4, 8.56, r"Parallel independent evaluation of target pair $(u, i)$, confidence scaling, score calibration, convex arbitration, and candidate filtering", 
            ha='left', va='center', fontsize=8.6, fontstyle='italic', color=text_muted)
    ax.plot([0.4, 16.8], [8.38, 8.38], color='#CBD5E1', lw=1.1)

    # -------------------------------------------------------------
    # 2. Input Column: Target Query (User u, Candidate Item i)
    # -------------------------------------------------------------
    draw_box(
        x=0.4, y=0.5, w=2.1, h=7.55,
        bg_color=c_query,
        border_color=border_slate,
        lw=1.3,
        title="TARGET QUERY",
        subtitle=r"Candidate Pair $(u, i)$",
        items=[
            (r"$\mathbf{Target \; User:} \; u$", True, False, text_dark, 8.2),
            (r"$\mathbf{Candidate:} \; i \in \mathcal{I} \setminus \mathcal{H}_u$", True, False, text_dark, 8.2),
            "",
            ("Collaborative Signal:", True, False, border_blue, 8.0),
            (r"Implicit play count $r_{ui}$", False, False, text_dark, 7.8),
            (r"$r_{ui} \in \mathbb{N}_{\geq 0}$", False, False, text_muted, 7.5),
            "",
            ("Semantic Signal:", True, False, border_green, 8.0),
            ("Artist social tags", False, False, text_dark, 7.8),
            (r"HetRec $\mathcal{T}_i \text{ and } \mathcal{H}_u$", False, False, text_muted, 7.5),
            "",
            ("Temporal Signal:", True, False, border_amber, 8.0),
            ("Interaction timestamps", False, False, text_dark, 7.8),
            (r"Age: $\Delta t_{ui} = (t_{\mathrm{ref}} - t_{ui})$", False, False, text_dark, 7.5),
            "",
            ("Independent Feeds", False, True, text_muted, 7.4),
            ("to 3 Parallel Streams", False, True, text_muted, 7.4)
        ]
    )

    # -------------------------------------------------------------
    # STREAM 1 — COLLABORATIVE (Top Track, y: 5.75 to 7.95)
    # -------------------------------------------------------------
    draw_stream_badge(2.9, 8.05, 7.9, 0.28, "STREAM 1 — COLLABORATIVE FILTERING (IMPLICIT MSVD / ALS)", '#DBEAFE', border_blue)

    # Box 1A: Confidence Transformation
    draw_box(
        x=2.9, y=5.75, w=2.45, h=2.2,
        bg_color=c_stream1,
        border_color=border_blue,
        title="Confidence Transform",
        subtitle="Implicit Feedback Scaling",
        items=[
            ("Binary preference:", False, True, text_muted, 7.6),
            (r"$p_{ui} = \mathbb{I}(r_{ui} > 0)$", False, False, text_dark, 8.0),
            "",
            ("Confidence weighting:", False, True, text_muted, 7.6),
            (r"$c_{ui} = 1 + \kappa \ln(1 + r_{ui})$", True, False, '#1E40AF', 8.2),
            (r"with $\kappa = 40.0$", False, False, text_dark, 7.8)
        ]
    )

    # Box 1B: Implicit MSVD / ALS
    draw_box(
        x=5.65, y=5.75, w=2.45, h=2.2,
        bg_color=c_stream1,
        border_color=border_blue,
        title="Implicit MSVD / ALS",
        subtitle="Latent Factor Projection",
        items=[
            (r"Factors: $K = 64, \quad \lambda = 0.05$", False, False, text_dark, 7.8),
            ("10 ALS iterations", False, False, text_dark, 7.8),
            ("", False, False, text_dark, 7.8),
            ("Raw Collaborative Score:", True, False, '#1E3A8A', 7.8),
            (r"$s_{ui}^{\mathrm{msvd}} = x_u^T y_i$", True, False, '#1E40AF', 8.6),
            (r"Latent vectors $x_u, y_i \in \mathbb{R}^{64}$", False, True, text_muted, 7.4)
        ]
    )

    # Box 1C: Per-User Min-Max Normalization
    draw_box(
        x=8.4, y=5.75, w=2.45, h=2.2,
        bg_color=c_stream1,
        border_color=border_blue,
        title="Per-User Normalization",
        subtitle="Candidate Score Calibration",
        items=[
            ("Min-max over candidate set:", False, True, text_muted, 7.6),
            (r"$\tilde{s}_{ui}^{\mathrm{msvd}} = \frac{s_{ui}^{\mathrm{msvd}} - \min_j s_{uj}^{\mathrm{msvd}}}{\max_j s_{uj}^{\mathrm{msvd}} - \min_j s_{uj}^{\mathrm{msvd}}}$", True, False, '#1E3A8A', 8.0),
            "",
            ("Normalized Score:", True, False, border_blue, 8.0),
            (r"$\tilde{s}_{ui}^{\mathrm{msvd}} \in [0, 1]$", True, False, '#1E40AF', 8.4)
        ]
    )

    # Connect Stream 1
    draw_arrow(2.5, 6.85, 2.9, 6.85, label=r"$r_{ui}$", color=border_blue)
    draw_arrow(5.35, 6.85, 5.65, 6.85, label=r"$c_{ui}$", color=border_blue)
    draw_arrow(8.1, 6.85, 8.4, 6.85, label=r"$s_{ui}^{\mathrm{msvd}}$", color=border_blue)

    # -------------------------------------------------------------
    # STREAM 2 — SEMANTIC (Middle Track, y: 3.15 to 5.35)
    # -------------------------------------------------------------
    draw_stream_badge(2.9, 5.42, 7.9, 0.28, "STREAM 2 — SEMANTIC TAG AFFINITY (TF-IDF ITEM & USER PROJECTION)", '#D1FAE5', border_green)

    # Box 2A: 2,299-tag Vocabulary & Item Representation
    draw_box(
        x=2.9, y=3.15, w=2.45, h=2.2,
        bg_color=c_stream2,
        border_color=border_green,
        title="TF-IDF Item Vector",
        subtitle="HetRec Social Tag Space",
        items=[
            (r"Vocabulary: $|\mathcal{V}| = 2{,}299$", True, False, border_green, 8.0),
            (r"Frequency threshold: freq $\geq 5$", False, False, text_dark, 7.6),
            (r"$\mathrm{IDF}(t) = \ln \frac{1 + |\mathcal{I}|}{1 + \mathrm{DF}(t)} + 1$", False, False, text_dark, 7.6),
            ("Item representation:", False, True, text_muted, 7.5),
            (r"$t_i = \mathrm{TF\text{-}IDF}(i) / \|t_i\|_2$", True, False, '#065F46', 8.2)
        ]
    )

    # Box 2B: User Semantic Profile
    draw_box(
        x=5.65, y=3.15, w=2.45, h=2.2,
        bg_color=c_stream2,
        border_color=border_green,
        title="User Tag Profile",
        subtitle="Confidence-Weighted Projection",
        items=[
            ("Project user listening confidence", False, False, text_dark, 7.6),
            (r"vector $C_u$ onto tag matrix $T$:", False, False, text_dark, 7.6),
            ("", False, False, text_dark, 7.6),
            (r"$z_u = \mathrm{normalized}(C_u T)$", True, False, '#065F46', 8.2),
            (r"$z_u = \frac{C_u T}{\|C_u T\|_2} \in \mathbb{R}^{2{,}299}$", False, False, text_dark, 7.8),
            ("Dense user preference profile", False, True, text_muted, 7.4)
        ]
    )

    # Box 2C: Semantic Similarity
    draw_box(
        x=8.4, y=3.15, w=2.45, h=2.2,
        bg_color=c_stream2,
        border_color=border_green,
        title="Semantic Similarity",
        subtitle="Cosine Similarity Scoring",
        items=[
            ("Inner product of unit vectors:", False, True, text_muted, 7.6),
            ("Raw Dot-Product Similarity:", True, False, '#064E3B', 7.8),
            (r"$s_{ui}^{\mathrm{tag}} = z_u^T t_i$", True, False, '#065F46', 8.6),
            "",
            ("Normalized Score (Bounded):", True, False, border_green, 8.0),
            (r"$\tilde{s}_{ui}^{\mathrm{tag}} = s_{ui}^{\mathrm{tag}} \in [0, 1]$", True, False, '#064E3B', 8.4)
        ]
    )

    # Connect Stream 2
    draw_arrow(2.5, 4.25, 2.9, 4.25, label=r"$\mathcal{T}_i$", color=border_green)
    draw_arrow(5.35, 4.25, 5.65, 4.25, label=r"$t_i$", color=border_green)
    draw_arrow(8.1, 4.25, 8.4, 4.25, label=r"$z_u$", color=border_green)

    # -------------------------------------------------------------
    # STREAM 3 — TEMPORAL RECENCY (Bottom Track, y: 0.50 to 2.70)
    # -------------------------------------------------------------
    draw_stream_badge(2.9, 2.78, 7.9, 0.28, "STREAM 3 — TEMPORAL RECENCY PREFERENCE MODELING", '#FEF3C7', border_amber)

    # Box 3A: Exponential Decay Kernel
    draw_box(
        x=2.9, y=0.50, w=2.45, h=2.2,
        bg_color=c_stream3,
        border_color=border_amber,
        title="Interaction Decay",
        subtitle="Exponential Decay Kernel",
        items=[
            ("Interaction elapsed days:", False, True, text_muted, 7.6),
            (r"$\Delta t_{ui} = (t_{\mathrm{ref}} - t_{ui}) / 86{,}400$", False, False, text_dark, 7.8),
            (r"Decay parameter: $\gamma = 0.10 \text{ day}^{-1}$", False, False, text_dark, 7.8),
            (r"Half-life $t_{1/2} = \ln 2 / \gamma = 6.93$ days", False, True, text_muted, 7.4),
            (r"Raw Score: $S_{ui} = c_{ui} \exp(-\gamma \Delta t_{ui})$", True, False, '#92400E', 8.0)
        ]
    )

    # Box 3B: User Recency Vector Assembly
    draw_box(
        x=5.65, y=0.50, w=2.45, h=2.2,
        bg_color=c_stream3,
        border_color=border_amber,
        title="Temporal Profile",
        subtitle="Interaction Vector Assembly",
        items=[
            ("Aggregate decayed affinity", False, False, text_dark, 7.6),
            ("across historical plays:", False, False, text_dark, 7.6),
            ("", False, False, text_dark, 7.6),
            (r"$S_u^{\mathrm{temp}} \in \mathbb{R}^{|\mathcal{I}|}$", True, False, text_dark, 8.4),
            ("Sparse CSR representation", False, True, text_muted, 7.5),
            ("Non-recurrent point decay", False, True, text_muted, 7.4)
        ]
    )

    # Box 3C: Normalization
    draw_box(
        x=8.4, y=0.50, w=2.45, h=2.2,
        bg_color=c_stream3,
        border_color=border_amber,
        title="Score Normalization",
        subtitle="Per-User Unit L2 Scaling",
        items=[
            ("Vector L2 normalization:", False, True, text_muted, 7.6),
            (r"$\tilde{s}_u^{\mathrm{temporal}} = \frac{S_u^{\mathrm{temp}}}{\|S_u^{\mathrm{temp}}\|_2}$", True, False, '#78350F', 8.2),
            "",
            ("Normalized Score:", True, False, border_amber, 8.0),
            (r"$\tilde{s}_{ui}^{\mathrm{temporal}} \in [0, 1]$", True, False, '#92400E', 8.4)
        ]
    )

    # Connect Stream 3
    draw_arrow(2.5, 1.60, 2.9, 1.60, label=r"$\Delta t_{ui}$", color=border_amber)
    draw_arrow(5.35, 1.60, 5.65, 1.60, label=r"$S_{ui}$", color=border_amber)
    draw_arrow(8.1, 1.60, 8.4, 1.60, label=r"$S_u$", color=border_amber)

    # -------------------------------------------------------------
    # 4. Convergence: Global Normalized Multi-Stream Arbiter
    # -------------------------------------------------------------
    draw_box(
        x=11.6, y=1.2, w=2.5, h=6.5,
        bg_color=c_fusion,
        border_color=border_red,
        lw=1.6,
        title="GLOBAL NORMALIZED",
        subtitle="MULTI-STREAM ARBITER",
        items=[
            ("Convex Linear Fusion", True, False, border_red, 8.4),
            (r"$\sum_{m} w_m = 1.00, \quad w_m \geq 0$", False, True, text_muted, 7.6),
            ("", False, False, text_dark, 7.0),
            ("Fixed Blending Weights:", True, False, text_dark, 7.8),
            (r"$w_{\mathrm{msvd}} = \mathbf{0.20}$", True, False, border_blue, 8.2),
            (r"$w_{\mathrm{tag}} = \mathbf{0.30}$", True, False, border_green, 8.2),
            (r"$w_{\mathrm{temp}} = \mathbf{0.50}$", True, False, border_amber, 8.2),
            ("", False, False, text_dark, 7.0),
            ("Final Scoring Equation:", True, False, text_dark, 8.2),
            (r"$\hat{r}_{ui} = 0.20 \, \tilde{s}_{ui}^{\mathrm{msvd}}$", True, False, '#991B1B', 8.4),
            (r"$+ \; 0.30 \, \tilde{s}_{ui}^{\mathrm{tag}}$", True, False, '#991B1B', 8.4),
            (r"$+ \; 0.50 \, \tilde{s}_{ui}^{\mathrm{temporal}}$", True, False, '#991B1B', 8.4),
            ("", False, False, text_dark, 7.0),
            ("Independent Stream Scoring", False, True, text_muted, 7.4),
            ("Validation Grid Optimized", False, True, text_muted, 7.4)
        ]
    )

    # Convergence Arrows from the 3 streams into the Arbiter
    # Collaborative incoming arrow
    draw_arrow(10.85, 6.85, 11.6, 5.65, label=r"$w_{\mathrm{msvd}} = 0.20$", rad=0.08, color=border_blue, 
               label_offset=(0, 0.16), badge_bg='#EFF6FF', badge_edge=border_blue, lw=1.5)
    # Semantic incoming arrow
    draw_arrow(10.85, 4.25, 11.6, 4.45, label=r"$w_{\mathrm{tag}} = 0.30$", rad=0.0, color=border_green, 
               label_offset=(0, 0.14), badge_bg='#F0FDF4', badge_edge=border_green, lw=1.5)
    # Temporal incoming arrow
    draw_arrow(10.85, 1.60, 11.6, 3.25, label=r"$w_{\mathrm{temp}} = 0.50$", rad=-0.08, color=border_amber, 
               label_offset=(0, -0.16), badge_bg='#FFFBEB', badge_edge=border_amber, lw=1.5)

    # -------------------------------------------------------------
    # 5. Output Stages: Candidate Filtering & Top-K Recommendation
    # -------------------------------------------------------------
    # Candidate Filtering Box
    draw_box(
        x=14.8, y=4.75, w=2.0, h=2.95,
        bg_color=c_filter,
        border_color=border_purple,
        lw=1.3,
        title="CANDIDATE FILTERING",
        subtitle=r"Exclude History $\mathcal{H}_u$",
        items=[
            ("Consumption History:", True, False, text_dark, 7.8),
            (r"$\mathcal{H}_u = \mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}}$", False, False, text_dark, 7.6),
            "",
            ("Hard Score Masking:", True, False, border_purple, 7.8),
            (r"$\hat{r}_{ui} \leftarrow -\infty \quad \forall i \in \mathcal{H}_u$", True, False, '#6D28D9', 8.2),
            "",
            ("Guarantees zero repeat leakage", False, True, text_muted, 7.4),
            (r"Only unconsumed $i \notin \mathcal{H}_u$", False, True, text_muted, 7.4)
        ]
    )

    # Top-K Recommendation Box
    draw_box(
        x=14.8, y=1.2, w=2.0, h=3.0,
        bg_color=c_out,
        border_color=border_mint,
        lw=1.5,
        title="TOP-K RECOMMENDATION",
        subtitle="Catalog-Scale Ranking",
        items=[
            ("Candidate Catalog:", True, False, text_dark, 7.8),
            (r"$|\mathcal{I}| = 1{,}493{,}688$ tracks", False, False, text_dark, 7.6),
            "",
            ("Partial Sorting:", True, False, text_dark, 7.8),
            (r"$\operatorname{arg\,max}^{(K)}_{i \notin \mathcal{H}_u} \hat{r}_{ui}$", True, False, '#047857', 8.2),
            ("numpy.argpartition", False, True, text_muted, 7.5),
            "",
            (r"Output: $\mathrm{Top\text{-}}K(u)$", True, False, '#065F46', 8.6),
            ("Latency: 96.48 ms/user", False, True, text_muted, 7.4)
        ]
    )

    # Output Arrows
    # Arbiter to Candidate Filtering: horizontal clean arrow
    draw_arrow(14.1, 6.22, 14.8, 6.22, label=r"Final Score $\hat{r}_{ui}$", rad=0.0, color=border_red, lw=1.4, label_offset=(0, 0.16))
    # Candidate Filtering to Top-K: vertical clean arrow
    draw_arrow(15.8, 4.75, 15.8, 4.2, label=r"Filtered Catalog", color=border_purple, lw=1.4, label_offset=(0.42, 0.0))

    # Save outputs
    pdf_path = "paper/mathematical_scoring_pipeline.pdf"
    svg_path = "paper/mathematical_scoring_pipeline.svg"
    png_path = "paper/mathematical_scoring_pipeline.png"

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Successfully exported diagrams to:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == '__main__':
    generate_scoring_pipeline_diagram()
