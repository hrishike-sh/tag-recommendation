# Phase 7 — Temporal & Nostalgia Recommendation Extension (Stream B)

## 1. Problem Formulation & Motivation

The standard implicit MSVD model computes user preferences collapsed across the entire historical observation window:
$$\hat{r}_{ui}^{\text{MSVD}} = \mathbf{x}_u^T \mathbf{y}_i$$
However, music listening displays strong temporal non-stationarity:

1. **Recency Dynamics:** Users develop short-term passions or trending artist affinities that dominate current listening sessions.
2. **Nostalgia Dynamics:** Users retain historical affinities for artists or tracks they loved in earlier years, occasionally rediscovering dormant preferences.
3. **Persistent Relevance:** Core taste preferences persist steadily across time.

Phase 7 integrates **Stream B (Temporal / Nostalgia Preference)** into the recommendation score alongside collaborative latent factors (MSVD) and semantic tag affinities (Stream A).

---

## 2. Mathematical Definition of Temporal Affinity

Let $\mathcal{H}_u = \{ (i, a(i), t) \}$ denote the chronological sequence of listening events for user $u$ strictly prior to the evaluation cutoff $T_{\text{ref}}$ ($t < T_{\text{ref}}$).

### A. Temporal Age and Decay Kernel

For any historical event at UTC timestamp $t$, its age in days relative to the reference cutoff $T_{\text{ref}}$ is:
$$\Delta t = \frac{T_{\text{ref}} - t}{86400 \text{ seconds}} \ge 0$$

The temporal weight of this interaction under decay rate $\gamma \ge 0$ is governed by an exponential decay kernel:
$$w(t; \gamma, \text{mode}) = \begin{cases}
\exp(-\gamma \cdot \Delta t), & \text{mode = 'recency'} \\
1.0 - \exp(-\gamma \cdot \Delta t), & \text{mode = 'nostalgia'} \\
1.0, & \gamma = 0 \text{ (uniform temporal weighting)}
\end{cases}$$

- In **Recency mode** ($\gamma > 0$), recent listening events ($\Delta t \to 0$) receive weights near $1.0$, while old events ($\Delta t \gg 0$) exponentially decay toward $0$.
- In **Nostalgia mode** ($\gamma > 0$), distant historical events receive higher weights ($1 - e^{-\gamma \Delta t} \to 1$), emphasizing dormant long-term taste over recent trends.

### B. Item & Artist Temporal Aggregation
Because unseen item evaluation requires recommending tracks the user has not previously consumed in training, track-level interaction decay alone cannot score unplayed tracks by familiar artists. We therefore construct two complementary temporal affinity matrices:

1. **User-Track Temporal Affinity $\mathbf{S}^{\text{track}} \in \mathbb{R}^{U \times I}$:**
   $$S_{ui}^{\text{track}} = \sum_{e \in \mathcal{H}_{u, i}} c_{ui} \cdot w(t_e; \gamma)$$
2. **User-Artist Temporal Affinity $\mathbf{S}^{\text{artist}} \in \mathbb{R}^{U \times A}$:**
   $$S_{ua}^{\text{artist}} = \sum_{e \in \mathcal{H}_{u, a}} c_{ua} \cdot w(t_e; \gamma)$$

For any candidate catalog track $i$ with parent artist $a(i)$, the combined temporal affinity is:
$$\text{Affinity}_{\text{temporal}}(u, i) = S_{ui}^{\text{track}} + \eta \cdot S_{u, a(i)}^{\text{artist}}$$
where $\eta \in [0, 1]$ (default $\eta = 1.0$) transfers user temporal artist affinity to all candidate tracks of that artist.

### C. Normalization
To prevent high-volume listeners from skewing recommendation score scales, the temporal affinity vector for user $u$ across all catalog items is $L_2$-normalized:
$$\hat{\mathbf{s}}_u = \begin{cases}
\frac{\mathbf{s}_u}{\|\mathbf{s}_u\|_2}, & \|\mathbf{s}_u\|_2 > 0 \\
\mathbf{0}, & \|\mathbf{s}_u\|_2 = 0
\end{cases}$$

---

## 3. Score Fusion Architecture

The full multi-stream recommendation score combines collaborative filtering, semantic tags, and temporal affinity:
$$\text{Score}_{\text{full}}(u, i) = \mathbf{x}_u^T \mathbf{y}_i + \beta \cdot \text{Sim}_{\text{tag}}(u, i) + \alpha \cdot \hat{s}_{ui}$$

### Ablation Configurations:
- **A. Popularity Baseline:** Frequency-based top-$K$.
- **B. Implicit MSVD:** $\beta = 0, \alpha = 0 \implies \mathbf{x}_u^T \mathbf{y}_i$.
- **C. Tag-Fused MSVD (Phase 6):** $\beta = 1.0, \alpha = 0 \implies \mathbf{x}_u^T \mathbf{y}_i + \beta (\hat{\mathbf{z}}_u^T \mathbf{t}_i)$.
- **D. Temporal-MSVD:** $\beta = 0, \alpha > 0 \implies \mathbf{x}_u^T \mathbf{y}_i + \alpha \hat{s}_{ui}$.
- **E. Tag + Temporal MSVD:** $\beta = 1.0, \alpha > 0 \implies \mathbf{x}_u^T \mathbf{y}_i + \beta (\hat{\mathbf{z}}_u^T \mathbf{t}_i) + \alpha \hat{s}_{ui}$.

---

## 4. Cold-Start and Unseen Item Dynamics
- For known collaborative tracks with no recent temporal signal, $\hat{s}_{ui} = 0$, defaulting to collaborative and tag scores.
- For unseen tracks from known artists, the artist affinity term $S_{u, a(i)}^{\text{artist}}$ enables temporal recommendations for new tracks by favored artists.
- For completely cold-start items (unseen tracks & unseen artists), $\hat{s}_{ui} = 0$ and $\mathbf{y}_i = \mathbf{0}$, gracefully falling back to tag affinity $\beta (\hat{\mathbf{z}}_u^T \mathbf{t}_i)$.

---

## 5. Computational Complexity & Sparsity
- The number of active user-artist pairs is bounded by $< 900,000$ (density $< 0.5\%$).
- $\mathbf{S}^{\text{artist}}$ and $\mathbf{S}^{\text{track}}$ are stored as sparse CSR matrices.
- Scoring time per user is $O(K \cdot I + |\text{NNZ}(\mathbf{s}_u)|)$, maintaining sub-millisecond evaluation latency per user.
