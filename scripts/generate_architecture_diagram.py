import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    # Set standard clean sans-serif font
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # Clean 16:9 canvas
    fig, ax = plt.subplots(figsize=(16.5, 9.2), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # Color palette - clean academic tones (slate, subtle pastel fills, sharp borders)
    c_data = '#F8FAFC'        # light slate
    c_stream1 = '#F0F7FF'     # crisp soft blue
    c_stream2 = '#F2FDF5'     # crisp soft green
    c_stream3 = '#FFFDF0'     # crisp soft amber
    c_norm = '#F8FAFC'        # clean slate
    c_arbiter = '#FFF1F2'     # soft crimson/rose
    c_retrieval = '#F5F3FF'   # soft purple
    c_output = '#ECFDF5'      # soft emerald
    
    edge_color = '#334155'    # slate 700
    text_dark = '#0F172A'     # slate 900
    text_muted = '#475569'    # slate 600

    def draw_box(x, y, w, h, bg_color, title, subtitle="", items=None, border_color=edge_color, lw=1.2, radius=0.15):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=border_color,
            facecolor=bg_color,
            zorder=2
        )
        ax.add_patch(box)
        
        # Title and Subtitle
        if subtitle:
            ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_dark, zorder=3)
            ax.text(x + w/2, y + h - 0.52, subtitle, ha='center', va='center', fontsize=8.0, fontstyle='italic', color=text_muted, zorder=3)
            cur_y = y + h - 0.78
        else:
            ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_dark, zorder=3)
            cur_y = y + h - 0.65

        if items:
            for it in items:
                # Check for custom section headers inside items list
                if isinstance(it, tuple):
                    text, is_bold, is_italic, col = it
                    fw = 'bold' if is_bold else 'normal'
                    fs = 'italic' if is_italic else 'normal'
                    ax.text(x + w/2, cur_y, text, ha='center', va='center', fontsize=8.3, fontweight=fw, fontstyle=fs, color=col, zorder=3)
                else:
                    ax.text(x + w/2, cur_y, it, ha='center', va='center', fontsize=8.1, color=text_dark, zorder=3)
                cur_y -= 0.28

    def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, color='#475569', label_offset=(0, 0.12)):
        ax.annotate(
            "",
            xy=(x2, y2), xycoords='data',
            xytext=(x1, y1), textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.3,
                mutation_scale=12,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2)/2 + label_offset[0], (y1 + y2)/2 + label_offset[1]
            ax.text(mx, my, label, ha='center', va='center', fontsize=7.5, color=text_muted, zorder=5,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

    # -------------------------------------------------------------
    # Stage Column Headers
    # -------------------------------------------------------------
    headers = [
        (1.2, "DATA SOURCES", "Raw Logs & Semantic Corpus"),
        (4.35, "PARALLEL RETRIEVAL STREAMS", "Domain-Specific Scoring Channels"),
        (7.9, "PER-USER ALIGNMENT", "Dynamic Range Normalization"),
        (10.85, "MULTI-STREAM ARBITER", "Global Weighted Convex Fusion"),
        (14.25, "CANDIDATE SERVING", "Masking, Exact Retrieval & Top-K")
    ]
    for x_c, h_title, h_sub in headers:
        ax.text(x_c, 8.85, h_title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_dark)
        ax.text(x_c, 8.58, h_sub, ha='center', va='center', fontsize=8.0, fontstyle='italic', color=text_muted)
        ax.plot([x_c - 1.25, x_c + 1.25], [8.42, 8.42], color='#CBD5E1', lw=1.0)

    # -------------------------------------------------------------
    # Column 1: Data Sources
    # -------------------------------------------------------------
    draw_box(
        x=0.2, y=4.7, w=2.0, h=3.3,
        bg_color=c_data,
        title="Last.fm 1K Events",
        subtitle="Chronological Log Split",
        items=[
            "19,150,865 Events",
            "992 Users | 1.49M Items",
            r"Train: $t < \text{2009-04-01}$",
            r"Val: [2009-04, 2009-05)",
            r"Test: [2009-05, 2009-07)",
            ("Zero-Leakage Cutoff", False, True, text_muted)
        ]
    )

    draw_box(
        x=0.2, y=1.0, w=2.0, h=2.8,
        bg_color=c_data,
        title="HetRec 2011 Tags",
        subtitle="Artist-Level Metadata",
        items=[
            "2,299 Filtered Tags",
            "594,225 Items Covered",
            "39.78% Catalog Coverage",
            "11.92M Matrix NNZ",
            ("Sparse CSR TF-IDF", False, True, text_muted)
        ]
    )

    # -------------------------------------------------------------
    # Column 2: The Three Processing Streams
    # -------------------------------------------------------------
    # Stream 1: Collaborative
    draw_box(
        x=3.1, y=5.8, w=2.5, h=2.4,
        bg_color=c_stream1,
        title="STREAM 1: Collaborative",
        subtitle="Implicit MSVD (ALS)",
        items=[
            r"$c_{ui} = 1 + \kappa \ln(1 + r_{ui})$",
            r"Latent Factors: $K = 64$",
            r"$\lambda = 0.05, \, \kappa = 40.0$",
            "10 Full ALS Iterations",
            (r"Score: $s_{ui}^{\mathrm{msvd}} = x_u^T y_i$", True, False, text_dark)
        ]
    )

    # Stream 2: Semantic Tag
    draw_box(
        x=3.1, y=3.2, w=2.5, h=2.3,
        bg_color=c_stream2,
        title="STREAM 2: Semantic",
        subtitle="HetRec Tag TF-IDF",
        items=[
            r"Vocabulary: $|\mathcal{V}| = 2{,}299$",
            r"Track Vector: $t_i = \mathrm{TF\text{-}IDF}(i) / \|\cdot\|_2$",
            r"User Profile: $z_u = C_u T / \|C_u T\|_2$",
            (r"Score: $s_{ui}^{\mathrm{tag}} = z_u^T t_i$", True, False, text_dark)
        ]
    )

    # Stream 3: Temporal Recency
    draw_box(
        x=3.1, y=0.6, w=2.5, h=2.3,
        bg_color=c_stream3,
        title="STREAM 3: Temporal",
        subtitle="Temporal Recency Preference",
        items=[
            r"$\Delta t_{ui} = (t_{\mathrm{ref}} - t_{ui}) / 86{,}400$",
            r"Decay Rate: $\gamma = 0.10 \text{ day}^{-1}$",
            r"$S_{ui}^{\mathrm{temp}} = c_{ui} \exp(-\gamma \Delta t_{ui})$",
            (r"Score: $s_u^{\mathrm{temp}} = S_u^{\mathrm{temp}} / \|S_u^{\mathrm{temp}}\|_2$", True, False, text_dark)
        ]
    )

    # Data -> Streams routing
    draw_arrow(2.2, 6.7, 3.1, 7.0, label="Play Counts")
    draw_arrow(2.2, 5.8, 3.1, 4.4, label=r"$C_u$ Weights", rad=0.08)
    draw_arrow(2.2, 5.0, 3.1, 1.75, label="Timestamps", rad=-0.12)
    draw_arrow(2.2, 2.4, 3.1, 3.8, label="Tag Matrix T")

    # -------------------------------------------------------------
    # Column 3: Per-User Score Normalization
    # -------------------------------------------------------------
    draw_box(
        x=6.6, y=1.2, w=2.6, h=6.8,
        bg_color=c_norm,
        title="Score Alignment Layer",
        subtitle="Per-User Calibration",
        items=[
            ("", False, False, text_dark),
            ("Collaborative Min-Max:", True, False, text_dark),
            (r"$\tilde{s}_{ui}^{\mathrm{msvd}} = \frac{s_{ui}^{\mathrm{msvd}} - \min_j s_{uj}^{\mathrm{msvd}}}{\max_j s_{uj}^{\mathrm{msvd}} - \min_j s_{uj}^{\mathrm{msvd}}}$", False, False, text_dark),
            ("", False, False, text_dark),
            (r"Maps $(-\infty, \infty) \to [0, 1]$", False, True, text_muted),
            ("Resolves user score variance", False, False, text_dark),
            ("", False, False, text_dark),
            ("Auxiliary Stream Alignment:", True, False, text_dark),
            (r"Cosine Tag Score: $\tilde{s}_{ui}^{\mathrm{tag}} \in [0, 1]$", False, False, text_dark),
            (r"L2 Temporal: $\tilde{s}_{ui}^{\mathrm{temp}} \in [0, 1]$", False, False, text_dark),
            ("", False, False, text_dark),
            ("Cross-Stream Scale Compatibility", True, True, '#1E40AF')
        ]
    )

    # Streams -> Normalization routing
    draw_arrow(5.6, 7.0, 6.6, 6.2, label=r"$s_{ui}^{\mathrm{msvd}}$")
    draw_arrow(5.6, 4.35, 6.6, 4.35, label=r"$s_{ui}^{\mathrm{tag}}$")
    draw_arrow(5.6, 1.75, 6.6, 2.5, label=r"$s_{ui}^{\mathrm{temp}}$")

    # -------------------------------------------------------------
    # Column 4: Global Normalized Multi-Stream Arbiter
    # -------------------------------------------------------------
    draw_box(
        x=9.65, y=2.0, w=2.45, h=5.2,
        bg_color=c_arbiter,
        border_color='#DC2626',
        lw=1.6,
        title="Global Normalized Arbiter",
        subtitle="Multi-Stream Fusion",
        items=[
            ("Global Mixing Weights:", True, False, text_dark),
            (r"$w_{\mathrm{msvd}} = 0.20$", False, False, text_dark),
            (r"$w_{\mathrm{tag}} = 0.30$", False, False, text_dark),
            (r"$w_{\mathrm{temp}} = 0.50$", False, False, text_dark),
            ("", False, False, text_dark),
            ("Fused Scoring Function:", True, False, text_dark),
            (r"$\hat{r}_{ui} = 0.20 \, \tilde{s}_{ui}^{\mathrm{msvd}}$", False, False, text_dark),
            (r"$+ \; 0.30 \, \tilde{s}_{ui}^{\mathrm{tag}}$", False, False, text_dark),
            (r"$+ \; 0.50 \, \tilde{s}_{ui}^{\mathrm{temp}}$", False, False, text_dark),
            ("", False, False, text_dark),
            ("Validation Optimized", True, False, '#991B1B'),
            ("Weights strictly frozen", False, True, text_muted)
        ]
    )

    # Normalization -> Arbiter routing
    draw_arrow(9.2, 4.6, 9.65, 4.6, label="Aligned Streams")

    # -------------------------------------------------------------
    # Column 5: Candidate Filtering, Top-K Retrieval, Output
    # -------------------------------------------------------------
    # Step A: Candidate Filtering
    draw_box(
        x=12.9, y=6.2, w=2.7, h=1.8,
        bg_color=c_retrieval,
        title="Candidate Masking",
        subtitle="Seen Interaction Filter",
        items=[
            (r"Historical Set: $\mathcal{H}_u = \mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}}$", False, False, text_dark),
            (r"Masking: $\hat{r}_{ui} \leftarrow -\infty \quad \forall i \in \mathcal{H}_u$", False, False, text_dark),
            ("Pure Novel Discovery", True, True, text_muted)
        ]
    )

    # Step B: Exact Top-K
    draw_box(
        x=12.9, y=3.6, w=2.7, h=2.2,
        bg_color=c_retrieval,
        title="Exact Top-K Retrieval",
        subtitle="Full Catalog Ranking",
        items=[
            (r"Catalog: $|\mathcal{I}| = 1{,}493{,}688$ Items", True, False, text_dark),
            ("Exact numpy.argpartition", False, False, text_dark),
            ("No ANN / Graph Approximations", False, True, text_muted),
            ("Latency: 96.48 ms (10.36 qps)", False, False, text_dark),
            ("Sparse Memory Safe (0 MB Dense)", True, False, '#4C1D95')
        ]
    )

    # Step C: Top-K Output
    draw_box(
        x=12.9, y=0.7, w=2.7, h=2.5,
        bg_color=c_output,
        border_color='#059669',
        lw=1.5,
        title="Top-K Recommendations",
        subtitle="Ordered Ranking & Attribution",
        items=[
            (r"$\mathrm{Top\text{-}}K(u) = \operatorname{arg\,topK}_{i} \hat{r}_{ui}$", True, False, text_dark),
            (r"Evaluated at $K \in \{5, 10, 20\}$", False, False, text_dark),
            ("NDCG@10: 0.007414", True, False, '#047857'),
            ("Low-Activity Gain: +50.47%", True, False, '#047857'),
            ("", False, False, text_dark),
            (r"Linear Contribution Attribution:", False, True, text_muted),
            (r"$\mathrm{Contrib}_m(u,i) = \frac{w_m \tilde{s}_{ui}^m}{\hat{r}_{ui}} \times 100\%$", False, False, text_dark)
        ]
    )

    # Right-pipeline routing
    draw_arrow(12.1, 5.0, 12.9, 6.8, label="Score Matrix", rad=0.08, label_offset=(0, 0.15))
    draw_arrow(14.25, 6.2, 14.25, 5.8)
    draw_arrow(14.25, 3.6, 14.25, 3.2)

    plt.tight_layout()
    
    # Save formats
    pdf_path = "paper/architecture_overview.pdf"
    svg_path = "paper/architecture_overview.svg"
    png_path = "paper/architecture_overview.png"
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Exported clean diagram to:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == '__main__':
    generate_diagram()
