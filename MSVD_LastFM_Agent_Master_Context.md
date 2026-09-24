# MSVD + Last.fm Agent Master Context

## Mission

Build the research prototype and IEEE paper for an MSVD-based music recommender adapted from the existing Tag-Based MSVD project.

Final system:

1. implicit-feedback MSVD/ALS
2. Tag-Fused MSVD discovery stream
3. dormant-preference / nostalgia stream
4. alpha arbiter
5. chronological evaluation
6. reproducible figures
7. optional Go serving/ANN layer after correctness is established

The paper must exceed 10 IEEE conference pages through substantive technical material, experiments, figures and tables, not filler.

## Source hierarchy

- **Repository:** truth for what is actually implemented.
- **Part 6 Tag-Based MSVD:** source for TF-IDF, Jaccard, weighted tag similarity, cosine/content similarity, and fusion/re-ranking.
- **Project proposal:** source for Last.fm adaptation, Stream A, Stream B, dynamic arbiter and project goals.
- **Current paper/spec:** implementation proposal, NOT evidence. Any numbers/results in it are unverified until reproduced.

## Critical rules

Never invent:

- benchmark results
- latency
- EDA statistics
- optimal nostalgia cycle
- baseline performance
- real recommendation traces
- tag coverage
- FAISS/HNSW usage

Do not call generic ALS “the original MSVD” unless the source actually proves that. Describe it as the project's implementation of the implicit MSVD formulation when appropriate.

## Current draft problems

The current draft contains unverified results and placeholders:

- abstract claims empirical improvement
- results table contains numerical values that have not been measured
- qualitative Tool/Massive Attack/Portishead trace is hypothetical unless reproduced
- latency table is unverified
- 22% cyclicity claim is unverified
- 180-day cycle claim is unverified
- intro says exponential nostalgia while methodology uses Gaussian
- confidence equation may not match repository implementation
- tag-fusion equation differs from the original additive Part 6 formulation
- references need auditing

Remove/mark all of these until experiments provide evidence.

## Target architecture

Last.fm events
 -> cleaning + chronological split
 -> P_ui + C_ui
 -> implicit MSVD/ALS
 -> candidate generation
 -> Tag Fusion (Stream A)
 -> dormant retrieval (Stream B)
 -> score normalization
 -> alpha arbiter
 -> Top-K + explanation

## Stream A

Preference:
p_ui = 1 if r_ui > 0, else 0.

Confidence must match the repository's actual implementation. Use logarithmic scaling if that is what the repo implements. Do not silently replace it.

ALS objective:
L = sum c_ui (p_ui - x_u^T y_i)^2
    + lambda(sum ||x_u||^2 + sum ||y_i||^2)

Requirements:

- sparse implementation
- numerically stable linear solves, not explicit matrix inversion
- configurable factors, regularization, confidence scale, epochs, seed
- training logs
- save/load
- Top-N recommendation
- exclude consumed items by default

## Tag-Fused MSVD

Follow the Part 6 source first:

1. base MSVD candidates
2. item/content similarity
3. TF-IDF tag relevance
4. Jaccard tag overlap
5. weighted tag similarity
6. fusion
7. rerank

First reproduce the source formulation. If normalization/gamma weighting is added for Last.fm, make it an explicit extension and evaluate it separately.

Do not assume Last.fm 1K contains usable tags. If an external tag source is needed, document source, mapping, coverage, unmatched items and provenance.

## Stream B

Historical dormant items:

- historical play count
- last historical interaction
- dormancy duration
- configurable dormant threshold
- configurable historical-volume threshold

Candidate Gaussian score:
N(u,i) = log(1 + historical_plays)
         *exp(-(dormancy - mu)^2/(2*sigma^2))

Do not claim mu=180 days is empirically optimal until measured.

Recommended experiment:

- no temporal kernel
- exponential
- Gaussian

All Stream B data must be available before the evaluation cutoff.

## Arbiter

Normalize streams independently, then:
Final = alpha*A_norm + (1-alpha)*B_norm

Support alpha sweep, e.g. 0, .25, .5, .75, 1.

Preserve explanation metadata:

- stream ID
- anchor tag if Stream A
- dormancy days if Stream B

## Evaluation

Use chronological train/validation/test splits. No future leakage.

Metrics:

- Recall@K
- NDCG@K
- MAP@K where practical
- Intra-List Diversity
- Nostalgia Recall@K
- optionally catalog coverage, novelty and long-tail coverage

Baselines:

1. popularity
2. implicit ALS/base MSVD
3. Tag-Fused MSVD
4. full Dual-Stream
Optional only if actually implemented: Item-KNN, BPR, time-aware baseline.

Ablations:

- no tag fusion
- no nostalgia
- semantic-only
- exponential vs Gaussian vs none
- alpha sweep
- linear vs logarithmic confidence if practical

## Paper structure for >10 pages

1 Abstract
2 Introduction
3 Background and Related Work
4 Problem Definition + Research Questions
5 Dataset + Data Engineering
6 Proposed Method
7 System Architecture + Implementation
8 Experimental Methodology
9 Results
10 Ablation Study
11 Qualitative Analysis
12 Error Analysis
13 Discussion
14 Limitations + Threats to Validity
15 Conclusion + Future Work

Use substantive content: equations, algorithms, complexity analysis, dataset tables, figures, baselines, results and discussion.

## Figures

Generate with Python/matplotlib, preferably PDF for Overleaf:

- architecture
- temporal split
- sparsity
- interaction/play-count distribution
- tag coverage
- ALS convergence
- alpha sensitivity
- temporal-kernel comparison
- baseline comparison
- ablation
- long-tail/catalog coverage
- qualitative explanation trace

Figures must read actual experiment outputs; never hard-code fake results.

# Agent task prompts

## Task 1 — Audit repository

Inspect the repository only. Identify:

- P_ui representation
- C_ui representation
- mappings
- cutoff/splits
- sparse structures
- package structure
- tests/dependencies
- mismatch between repo and paper

Do not implement ALS. Report exact files and proposed model interface.

## Task 2 — Implement ALS/MSVD

Implement confidence-weighted implicit ALS using the existing data structures.
Requirements:

- sparse, stable solves
- configurable factors/reg/epochs/seed
- logging
- save/load
- recommendation API
- exclude consumed items
- unit tests on a tiny synthetic dataset

Do not implement tags, nostalgia, ANN or HTTP yet.
Run all tests and report changed files and commands.

## Task 3 — Evaluation harness

Implement chronological train/validation/test evaluation.
Metrics: Recall@K, NDCG@K, MAP@K if practical.
Handle users with no test positives.
Export CSV/JSON.
Create a synthetic test with known metric values.
No tags/nostalgia yet.

## Task 4 — Tag pipeline

Determine whether current Last.fm data has tags. If not, integrate a reproducible external tag source.
Report:

- identifier mapping
- matched/unmatched counts
- coverage
- normalized tags
- tag frequencies
- provenance
Do not fabricate tags.

## Task 5 — Tag Fusion

Implement:
candidate generation -> TF-IDF -> Jaccard -> weighted Jaccard -> fusion -> rerank.
First reproduce Part 6 formulation. Any Last.fm-specific normalization must be explicit and configurable.
Unit-test TF-IDF, Jaccard, weighted score and ranking.

## Task 6 — Dormant retrieval

Implement historical volume + last-play + dormancy + configurable thresholds + Gaussian kernel.
Parameters configurable; do not claim 180 days as validated.
Unit-test synthetic timestamps.

## Task 7 — Arbiter

Implement independent score normalization and:
Final = alpha*A + (1-alpha)*B.
Handle overlapping/disjoint candidate sets.
Preserve explanations.
Test alpha=0/1, overlap, disjoint, ties and missing explanations.

## Task 8 — End-to-end runner

One reproducible command:
data -> preprocessing -> training -> candidates -> tags -> dormant -> arbiter -> evaluation.
Record seed, parameters, dataset/cutoff, model metadata and outputs.
Never overwrite runs.

## Task 9 — Figure generator

Create Python scripts for all paper figures.
Read values from experiment outputs.
Save PDF + PNG.
No ASCII diagrams.
No hard-coded experimental values.

## Task 10 — Go serving layer

Only after Python research pipeline is correct.
Load persisted model/index, serve Top-K + explanations, benchmark latency.
Do not claim FAISS/HNSW unless implemented and benchmarked.
If ANN is added, compare exact-search recall and latency.

## Paper-writing agent

Revise only from verified code/results.
Expand related work, problem definition, data engineering, math, algorithms, complexity, architecture, experiment protocol, baselines, metrics, results, ablations, qualitative/error analysis, discussion and limitations.
No filler.
Every number must come from an experiment artifact.
Every equation must match code.
Clearly distinguish inherited Tag-MSVD, music adaptation, nostalgia contribution and engineering choices.

## Definition of done

- ALS/MSVD reproducible
- chronological evaluation verified
- tags integrated with measured coverage
- tag fusion tested
- dormant engine tested
- arbiter tested
- baselines run fairly
- ablations run
- metrics exported
- figures generated from outputs
- no fabricated numbers
- references audited
- equations match code
- end-to-end run reproducible
