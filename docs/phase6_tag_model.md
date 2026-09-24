# Phase 6 — Tag-Enhanced MSVD Mathematical Formulation

## 1. Item Tag Representation ($\mathbf{t}_i$)
Let $\mathcal{T}$ denote the vocabulary of $T$ retained distinct tags.
For each track $i \in \{0, \dots, I-1\}$, let $a(i)$ denote its parent artist.
The raw tag count of tag $t \in \mathcal{T}$ for artist $a$ is:
$$TF(t, a) = \sum_{u \in \text{HetRec}} \mathbb{I}(\text{user } u \text{ tagged artist } a \text{ with } t)$$

The Inverse Document Frequency ($IDF$) of tag $t$ is calculated strictly across the training catalog $\mathcal{I}_{\text{train}}$:
$$IDF(t) = \ln \left( \frac{1 + |\mathcal{I}_{\text{train}}|}{1 + |\{ i \in \mathcal{I}_{\text{train}} : TF(t, a(i)) > 0 \}|} \right) + 1$$

The item tag vector $\mathbf{t}_i \in \mathbb{R}^T$ is defined as the $L_2$-normalized TF-IDF vector:
$$\mathbf{t}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2}, \quad \text{where } v_{i, t} = TF(t, a(i)) \cdot IDF(t)$$
If artist $a(i)$ has no tag data in HetRec, $\mathbf{t}_i = \mathbf{0}$.

---

## 2. User Tag Preference Profile ($\mathbf{z}_u$)
A user's semantic taste vector $\mathbf{z}_u \in \mathbb{R}^T$ aggregates the tag profiles of items they interacted with in the training period, weighted by their implicit confidence delta $d_{ui} = \kappa \ln(1 + n_{ui})$:
$$\mathbf{z}_u = \sum_{i \in \mathcal{R}_u} (1 + d_{ui}) \mathbf{t}_i = \sum_{i \in \mathcal{R}_u} c_{ui} \mathbf{t}_i$$
To ensure scale-invariance across heavy and light listeners, $\mathbf{z}_u$ is $L_2$-normalized:
$$\hat{\mathbf{z}}_u = \begin{cases} \frac{\mathbf{z}_u}{\|\mathbf{z}_u\|_2}, & \|\mathbf{z}_u\|_2 > 0 \\ \mathbf{0}, & \|\mathbf{z}_u\|_2 = 0 \end{cases}$$

---

## 3. Semantic Similarity Function
The semantic similarity between user $u$ and candidate item $i$ is the cosine similarity between their normalized tag profiles:
$$\text{Sim}_{\text{tag}}(u, i) = \hat{\mathbf{z}}_u^T \mathbf{t}_i \in [0, 1]$$

---

## 4. Score Fusion (Tag-Fused MSVD / Stream A)
The final recommendation score combines the base collaborative MSVD prediction $\hat{r}_{ui}^{\text{MSVD}} = \mathbf{x}_u^T \mathbf{y}_i$ with the semantic tag similarity:
$$\text{Score}_{\text{fused}}(u, i) = \hat{r}_{ui}^{\text{MSVD}} + \beta \cdot \text{Sim}_{\text{tag}}(u, i)$$
where $\beta \ge 0$ is the semantic fusion hyperparameter tuned strictly on the validation period.

- When $\beta = 0$, the formulation identically recovers pure implicit MSVD.
- For items with missing tags ($\mathbf{t}_i = \mathbf{0}$), $\text{Sim}_{\text{tag}}(u, i) = 0$, gracefully falling back to pure collaborative filtering without penalty.
- For cold-start items with $\mathbf{y}_i = \mathbf{0}$, $\text{Score}_{\text{fused}}(u, i) = \beta \cdot \text{Sim}_{\text{tag}}(u, i)$, enabling cold-start discovery for tagged tracks.
