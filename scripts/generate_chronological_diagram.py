import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_chronological_diagram():
    # Set standard clean academic sans-serif typography
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # IEEE two-column width friendly aspect (16.5 x 8.8 inches)
    fig, ax = plt.subplots(figsize=(16.5, 8.8), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 8.8)
    ax.axis('off')

    # Color palette - subtle academic shades
    c_raw = '#F8FAFC'         # slate 50
    c_train = '#EFF6FF'       # soft blue
    c_val = '#FEF3C7'         # soft amber
    c_test = '#ECFDF5'        # soft emerald
    c_meta = '#FAF5FF'        # soft purple
    c_catalog = '#F1F5F9'     # slate 100
    
    edge_color = '#334155'    # slate 700
    border_blue = '#2563EB'   # blue 600
    border_amber = '#D97706'  # amber 600
    border_green = '#059669'  # emerald 600
    border_purple = '#7C3AED' # purple 600
    
    text_dark = '#0F172A'     # slate 900
    text_muted = '#475569'    # slate 600

    def draw_box(x, y, w, h, bg_color, title, subtitle="", items=None, border_color=edge_color, lw=1.2, radius=0.12):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=border_color,
            facecolor=bg_color,
            zorder=2
        )
        ax.add_patch(box)
        
        if subtitle:
            ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_dark, zorder=3)
            ax.text(x + w/2, y + h - 0.52, subtitle, ha='center', va='center', fontsize=8.0, fontstyle='italic', color=text_muted, zorder=3)
            cur_y = y + h - 0.78
        else:
            ax.text(x + w/2, y + h - 0.32, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_dark, zorder=3)
            cur_y = y + h - 0.62

        if items:
            for it in items:
                if isinstance(it, tuple):
                    text, is_bold, is_italic, col = it
                    fw = 'bold' if is_bold else 'normal'
                    fs = 'italic' if is_italic else 'normal'
                    ax.text(x + w/2, cur_y, text, ha='center', va='center', fontsize=8.2, fontweight=fw, fontstyle=fs, color=col, zorder=3)
                else:
                    ax.text(x + w/2, cur_y, it, ha='center', va='center', fontsize=8.1, color=text_dark, zorder=3)
                cur_y -= 0.27

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
    # Section Headers
    # -------------------------------------------------------------
    ax.text(0.4, 8.4, "CHRONOLOGICAL ZERO-LEAKAGE EVALUATION PROTOCOL & METADATA PREPARATION", 
            ha='left', va='center', fontsize=12.0, fontweight='bold', color=text_dark)
    ax.text(0.4, 8.12, "Partitioning of 19.15M Last.fm listening logs and HetRec 2011 social tags over a 1.49M-item candidate catalog", 
            ha='left', va='center', fontsize=8.5, fontstyle='italic', color=text_muted)
    ax.plot([0.4, 16.1], [7.95, 7.95], color='#CBD5E1', lw=1.0)

    # -------------------------------------------------------------
    # Top Branch: Raw Listening Logs & Chronological Timeline
    # -------------------------------------------------------------
    # 1. Raw Data Source Box
    draw_box(
        x=0.4, y=4.5, w=2.6, h=3.0,
        bg_color=c_raw,
        border_color='#475569',
        title="RAW EVENT DATA",
        subtitle="Last.fm 1K Listening Logs",
        items=[
            ("19,150,865 Clean Events", True, False, text_dark),
            ("992 Clean Users", False, False, text_dark),
            ("1,505,185 Unique Tracks", False, False, text_dark),
            "",
            ("Implicit play counts:", False, True, text_muted),
            (r"$r_{ui} = \sum \mathbb{I}(u \text{ played } i)$", False, False, text_dark)
        ]
    )

    # 2. Time Axis Arrow
    ax.annotate(
        "",
        xy=(16.1, 7.35), xycoords='data',
        xytext=(3.4, 7.35), textcoords='data',
        arrowprops=dict(
            arrowstyle="->",
            color='#1E293B',
            lw=2.0,
            mutation_scale=15
        ),
        zorder=3
    )
    ax.text(3.5, 7.6, "CHRONOLOGICAL TIME AXIS", fontsize=9.0, fontweight='bold', color='#1E293B')
    ax.text(15.9, 7.6, "TIME t", ha='right', fontsize=9.0, fontweight='bold', color='#1E293B')

    # Arrow from Raw Data to Timeline
    draw_arrow(3.0, 6.0, 3.4, 6.0, label="Chronological Split")

    # 3. Train Partition Block
    draw_box(
        x=3.5, y=4.2, w=3.7, h=2.85,
        bg_color=c_train,
        border_color=border_blue,
        lw=1.5,
        title="TRAINING WINDOW",
        subtitle=r"$t < \text{2009-04-01T00:00:00Z}$",
        items=[
            ("Historical Listening Events", True, False, border_blue),
            ("All events prior to April 1, 2009", False, False, text_dark),
            (r"$\mathbf{Max \; Timestamp:}$", True, False, text_dark),
            ("2009-03-31 23:59:57", False, False, '#1E3A8A'),
            "",
            ("Model Parameters Fitted Strictly Here", False, True, text_muted),
            ("Confidence: c_ui = 1 + 40 ln(1 + r_ui)", False, False, text_dark)
        ]
    )

    # 4. Validation Partition Block
    draw_box(
        x=7.6, y=4.2, w=3.8, h=2.85,
        bg_color=c_val,
        border_color=border_amber,
        lw=1.5,
        title="VALIDATION WINDOW",
        subtitle=r"$t \in [\text{2009-04-01}, \, \text{2009-05-01})$",
        items=[
            ("Hyperparameter Tuning Partition", True, False, border_amber),
            ("1 Month Chronological Window", False, False, text_dark),
            (r"$\mathbf{Min \; Timestamp:}$ 2009-04-01 00:00:03", False, False, '#78350F'),
            (r"$\mathbf{Max \; Timestamp:}$ 2009-04-30 23:59:57", False, False, '#78350F'),
            "",
            ("Weight Optimization Grid", True, False, text_dark),
            ("Selects: w_msvd=0.2, w_tag=0.3, w_temp=0.5", False, False, text_dark)
        ]
    )

    # 5. Test Partition Block
    draw_box(
        x=11.8, y=4.2, w=4.3, h=2.85,
        bg_color=c_test,
        border_color=border_green,
        lw=1.5,
        title="TEST EVALUATION WINDOW",
        subtitle=r"$t \in [\text{2009-05-01}, \, \text{2009-07-01})$",
        items=[
            ("Final Out-of-Sample Evaluation", True, False, border_green),
            ("2 Months Evaluation Window", False, False, text_dark),
            (r"$\mathbf{Min \; Timestamp:}$ 2009-05-01 00:00:00", True, False, '#064E3B'),
            (r"$\mathbf{518 \; Evaluable \; Test \; Users}$ (>=1 interaction)", False, False, text_dark),
            "",
            ("Full Catalog Ranking: 1,493,688 Tracks", True, False, text_dark),
            ("Masking: Seen train/val items removed", False, True, text_muted)
        ]
    )

    # Strict Leakage Guardrail Banner
    leak_box = patches.FancyBboxPatch(
        (3.5, 3.55), 12.6, 0.45,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=1.2,
        edgecolor='#DC2626',
        facecolor='#FEF2F2',
        zorder=3
    )
    ax.add_patch(leak_box)
    ax.text(9.8, 3.77, r"NO TEMPORAL LEAKAGE:   $\max(\text{Train}) < \min(\text{Validation}) < \max(\text{Validation}) < \min(\text{Test})$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color='#B91C1C', zorder=4)

    # -------------------------------------------------------------
    # Bottom Branch: Metadata Processing & Final Catalog Alignment
    # -------------------------------------------------------------
    # Metadata Source
    draw_box(
        x=0.4, y=0.5, w=2.6, h=2.65,
        bg_color=c_meta,
        border_color=border_purple,
        title="HETREC 2011 TAGS",
        subtitle="Artist-Level Metadata",
        items=[
            ("Social Tag Assignments", True, False, border_purple),
            ("User-annotated artist tags", False, False, text_dark),
            ("Inherited by all artist tracks", False, True, text_muted),
            "",
            ("Frequency Filtering:", True, False, text_dark),
            (r"Minimum threshold: $\mathrm{freq} \geq 5$", False, False, text_dark)
        ]
    )

    # Filtered Tag Vocabulary
    draw_box(
        x=3.5, y=0.5, w=2.7, h=2.65,
        bg_color=c_meta,
        border_color=border_purple,
        title="FILTERED VOCABULARY",
        subtitle=r"$|\mathcal{V}| = 2{,}299 \text{ Tags}$",
        items=[
            ("Vocabulary Pruning", True, False, border_purple),
            ("Eliminates rare / noisy tags", False, False, text_dark),
            ("IDF computed strictly from", False, False, text_dark),
            ("historical catalog observations", False, False, text_dark),
            "",
            (r"$\text{IDF}(t) = \ln \frac{1 + |\mathcal{I}|}{1 + \text{DF}(t)} + 1$", False, False, text_dark)
        ]
    )

    # Sparse Item-Tag TF-IDF
    draw_box(
        x=6.6, y=0.5, w=2.9, h=2.65,
        bg_color=c_meta,
        border_color=border_purple,
        title="SPARSE TF-IDF MATRIX",
        subtitle="Item Representations",
        items=[
            ("Sparse CSR Tag Matrix T", True, False, border_purple),
            ("Shape: 1,493,688 x 2,299", False, False, text_dark),
            ("Total NNZ: 11,920,822", True, False, text_dark),
            "",
            ("Normalized track vectors:", False, True, text_muted),
            (r"$t_i = \mathrm{TF\text{-}IDF}(i) / \|\cdot\|_2$", False, False, text_dark),
            ("Zero dense memory overhead", False, True, text_muted)
        ]
    )

    # Final Candidate Catalog Alignment
    draw_box(
        x=10.0, y=0.5, w=6.1, h=2.65,
        bg_color=c_catalog,
        border_color='#334155',
        lw=1.5,
        title="FINAL EVALUATED CANDIDATE CATALOG ALIGNMENT",
        subtitle="Catalog Dimensions & Semantic Tag Coverage Realism",
        items=[
            (r"$\mathbf{Candidate \; Recommendation \; Catalog:}$   $|\mathcal{I}| = 1{,}493{,}688 \text{ unique tracks}$", True, False, text_dark),
            (r"$\mathbf{Tag\text{-}Covered \; Catalog \; Items:}$   $594{,}225 \text{ tracks} \; (\mathbf{39.7824\%} \text{ of full catalog})$", True, False, '#1E40AF'),
            ("Uncovered tracks (60.22%) receive zero tag mass; handled via collaborative and temporal streams", False, True, text_muted),
            "",
            (r"$\mathbf{Evaluation \; Integrity:}$ Pure novel discovery (seen training/val tracks masked per user)", True, False, text_dark),
            ("No zero-data cold-start assumption; non-annotated items naturally participate in collaborative ranking", False, False, text_muted)
        ]
    )

    # Metadata pipeline routing arrows
    draw_arrow(3.0, 1.82, 3.5, 1.82, label=r"freq $\geq$ 5")
    draw_arrow(6.2, 1.82, 6.6, 1.82, label="TF-IDF")
    draw_arrow(9.5, 1.82, 10.0, 1.82, label="Tag Mapping")

    # Connect Train Data to Catalog cleanly below the red banner
    draw_arrow(5.35, 4.2, 5.35, 3.25, rad=0.0)
    draw_arrow(5.35, 3.25, 10.0, 3.16, label="Observed Track Filter", rad=0.0, label_offset=(-2.0, 0.12))

    plt.tight_layout()

    # Save to outputs
    pdf_path = "paper/chronological_protocol.pdf"
    svg_path = "paper/chronological_protocol.svg"
    png_path = "paper/chronological_protocol.png"

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Exported chronological evaluation diagram to:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == '__main__':
    generate_chronological_diagram()
