# Phase 8 — Multi-Stream Recommendation Arbiter Audit & Score Analysis

## 1. Context & Architecture Overview
Phases 5, 6, and 7 established three distinct recommendation streams:
1. **Base Collaborative Filtering (MSVD):** Low-rank matrix factorization capturing global latent interaction preferences ($\hat{r}_{ui} = \mathbf{x}_u^T \mathbf{y}_i$).
2. **Stream A (Semantic Artist-Tag Similarity):** Sparse $L_2$-normalized TF-IDF artist genre vectors capturing semantic content preferences ($S_{\text{tag}}(u, i) = \hat{\mathbf{z}}_u^T \mathbf{t}_i \in [0, 1]$).
3. **Stream B (Temporal & Recency Affinity):** Exponentially decayed confidence-weighted historical listening intensity ($S_{\text{temp}}(u, i) = \hat{s}_{ui} \in [0, 1]$).

---

## 2. Empirical Scale & Range Analysis of Recommendation Streams
Examining score ranges and distributions across the 518 test users and 1,493,688 candidate items:

| Stream | Theoretical Range | Empirical User Min | Empirical User Max | Sparsity / Density | Characteristics |
|---|:---:|:---:|:---:|:---:|---|
| **Base MSVD ($\mathbf{x}_u^T \mathbf{y}_i$)** | $(-\infty, +\infty)$ | $\approx -0.05$ | $\approx +0.85$ | Dense ($100\%$) | Real-valued dot products; unnormalized scale across users. |
| **Stream A: Tags ($\hat{\mathbf{z}}_u^T \mathbf{t}_i$)** | $[0, 1]$ | $0.000$ | $\approx 0.65$ | Semi-sparse ($39.8\%$ item coverage) | Non-negative cosine similarities; zero for untagged items. |
| **Stream B: Temporal ($\hat{s}_{ui}$)** | $[0, 1]$ | $0.000$ | $\approx 0.70$ | Highly sparse ($0.3\%$ item coverage) | Non-negative decayed confidence; zero for unseen tracks. |

---

## 3. Why Naive Additive Fusion Underperforms
In Phase 7, fixed additive fusion $\text{Score} = \text{Score}_{\text{MSVD}} + 1.0 \cdot S_{\text{tag}} + 1.0 \cdot S_{\text{temp}}$ yielded mixed results:
- **Temporal signal dominates top ranks:** Because temporal affinity is non-zero only for tracks the user previously engaged with (or artist extensions), adding a raw temporal score directly can over-privilege recent tracks at the expense of novel collaborative discovery.
- **Tag signal aids tail ranking (@20):** Tag affinity provides broad thematic guidance for unplayed catalog items but is too diffuse to pinpoint the exact next track at top-5.
- **Heterogeneous User Behaviors:** Heavy users with broad listening histories have dense collaborative embeddings where MSVD is superior, whereas sparse/cold users rely heavily on temporal recency and semantic tags.

---

## 4. Phase 8 Arbiter Design Requirements
1. **Per-User Score Normalization:** Bring all three streams into a common $[0, 1]$ or $Z$-score distribution per user before combination.
2. **Adaptive Weighting Mechanisms:**
   - **Global Normalized Fusion:** Optimized baseline reference with normalized streams and simplex weights $\sum w_k = 1$.
   - **User-Aware Dynamic Gating:** User-specific weights $w(u) = \text{Softmax}(W f_u + b)$ parameterized by user interaction volume, profile span, and tag availability.
   - **Item-Aware Support Adjustment:** Modulating stream weights dynamically if an item lacks tag coverage or collaborative factors.
