# Stream A factorization and fusion; Stream B eligibility

Implemented September 22, 2026. This supersedes the future-work descriptions of
these components in the September 15 handover PDF. The PDF remains a historical
milestone record.

## What "implicit MSVD" means in this implementation

Stream A implements the proposal's staged pipeline: confidence-weighted latent
candidate generation followed by metadata/tag similarity reranking. The latent
core is weighted implicit ALS from `implicit.cpu.als`, not a truncated SVD of
`confidence * preference`. The earlier movie MSVD user/genre regularizers are not
specified in the supplied chapter. They are not invented or claimed as reproduced.
This is an explicit implicit-feedback adaptation, not a new factorization algorithm.

It optimizes the weighted squared-error objective in `data-contract.md`. Stored
positive interactions passed to ALS are **full confidence**
`1 + kappa * ln(1 + plays)`; missing entries imply `P=0, C=1`. Library `alpha=1`
avoids multiplying the confidence scale twice. Kappa comes from the matrix
snapshot and is not silently overridden. In particular, kappa=0 still preserves
observed preferences with confidence 1.

Default training: 32 factors, 10 ALS iterations, regularization 0.1, seed 42 and
four CPU threads. BLAS is limited to one thread to avoid nested oversubscription.
The solver uses conjugate-gradient ALS updates. Tests additionally compare exact
ALS item updates with dense normal equations including unobserved entries.

```powershell
uv sync --frozen --extra dev
uv run lastfm train --dataset 1k --factors 32 --iterations 10
uv run lastfm recommend --user-id user_000001 --k 10 --candidates 200
```

`--matrix-dir` selects a specific training snapshot. Otherwise the latest matrix
report supplies it. The trainer reads its immutable Parquet files, not the active
DuckDB views. Each model stores factors, counts, user/item mappings, matrix
metadata, primary artist metadata and training settings in `artifacts/models/`.
The published model report is `artifacts/reports/model-1k.json`.

Model folders are self-contained for discovery: the recorded source matrix path
is provenance, not required to recommend from an explicitly supplied `--model-dir`.
The matrix fingerprint binds its counts, mappings and cutoff to compatible tag
artifacts. Models and tag artifacts are never silently paired across snapshots.
Training loss is logged but is not a recommendation-quality metric.

## Candidate generation and content similarity

For one user, score the training catalog using the dot product of user and item
factors. Exclude every item the user played before the training cutoff. Return the
highest-scoring N candidates (200 by default), with deterministic index tie-breaks.
Only a single user's score vector is allocated, not a dense all-user score matrix.
Items outside the training catalog and unseen users currently have no cold-start
fallback. Unknown users cause an explicit error.

For fusion, choose the user's top 50 tracks by training play count. The content
profile is a `log1p(plays)`-weighted mean of their one-hot primary artist vectors.
Each candidate's artist cosine is its profile weight divided by the profile L2
norm, or zero when its artist is missing/not in that profile.

This is **artist metadata similarity, not acoustic similarity**. Primary artists
and display names are chosen deterministically from available source metadata
(lexicographic alias tie-break), restricted to training item IDs. This metadata
is treated as static; future listening counts never enter the content profile.
Artist identity alone is a limited content representation, to be extended when
actual audio/genre features are available. A different artist receives content
similarity zero, even when its sound is similar.

## Tags: required input and historical availability

The Last.fm 1K/360K archives do not contain the tag corpus. No real tag source was
available for this milestone. The importer and full fusion path are implemented
and integration-tested with explicitly synthetic fixtures. The prepared full-data
recommendations use ALS mode; their tag scores are null, not fabricated zero-evidence
matches. Requesting `--mode fusion` without `--tags-dir` fails explicitly.

Supply a UTF-8 TSV with a header and canonical **item IDs from items.parquet**:

```text
item_id<TAB>tag<TAB>count<TAB>assigned_at
```

`count` is a positive finite tag-assignment frequency, not the user's play count.
`assigned_at` is optional only when an explicit `--metadata-as-of` assertion supplies
the source's historical availability. A timestamp requires a timezone; DuckDB
normalizes parsed per-row timestamps to UTC. The global assertion must precede
the training cutoff. Rows at/after the cutoff are excluded. Untimestamped rows
without the assertion fail. An assertion is supplied evidence about the source,
not independent proof of historical availability: do not backdate current metadata.

```powershell
uv run lastfm tags --dataset 1k --source data/tags/item-tags.tsv
# Only for a source genuinely known to have existed at this historical time:
uv run lastfm tags --dataset 1k --source data/tags/item-tags.tsv --metadata-as-of 2009-04-30T00:00:00Z
uv run lastfm recommend --user-id user_000001 --tags-dir artifacts/tags/1k/ACTUAL_SNAPSHOT --mode fusion
```

Replace `ACTUAL_SNAPSHOT` with the output path printed by `tags`. The importer
normalizes case and whitespace, merges duplicate item/tag pairs, excludes unknown
item IDs, and records coverage and exclusion counts. Invalid frequencies, missing
identities and malformed timestamps fail the import. Artist-only tag sources must
first be mapped explicitly to canonical track IDs; no implicit name guessing occurs.

## TF-IDF and weighted overlap definitions

```text
TF(i,t) = tag_count(i,t) / sum_s tag_count(i,s)
IDF(t) = ln(N / df(t))
raw_relevance(i,t) = TF(i,t) * IDF(t)
r(i,t) = raw_relevance(i,t) / sum_s raw_relevance(i,s)
```

N is the full training item catalog, including untagged items. No smoothing is
applied. An ubiquitous tag has zero IDF. Zero-mass rows stay zero. Original tag
presence is preserved independently from relevance, including zero-IDF tags.
The raw TF-IDF and L1-normalized relevance are both exported as sparse matrices.
The user tag profile is the normalized weighted sum of favorite-item relevance,
with `log1p(plays)` weights and the union of those items' present tags.

Two metrics are available because the proposal's formula is not the standard
symmetric weighted Jaccard:

1. `--tag-metric proposal` (default): Jaccard of present-tag sets times the mean
   candidate relevance on shared tags. This is asymmetric, following the proposal.
2. `--tag-metric weighted-jaccard`: sum of minimum candidate/profile weights divided
   by sum of maximum weights. This is symmetric and equals zero for zero total mass.

Both return zero for disjoint tags or an empty profile. The default reranking is
`artist_cosine + tag_score`, with configurable nonnegative `--content-weight` and
`--tag-weight`. Scores are ranking values, not probabilities. ALS score then item
index break fusion ties. The candidate pool limits recall: a tag match outside
the ALS pool cannot be rescued by reranking.

Output includes the ALS score, content score, tag score, shared tags, missing-tag
indicator and ranking mode. These are auditable component explanations, not a
claim that an ALS latent factor semantically represents a specific tag.

## Stream B: dormancy extraction

```powershell
uv run lastfm dormant --dataset 1k --min-historical-plays 5
```

Eligibility is `historical_plays >= threshold AND active_plays = 0`. Historical
and active counts come from the chosen matrix snapshot, never later events.
The recent window is exactly `[cutoff - recent_days, cutoff)`. A play at the lower
boundary is active and disqualifies dormancy. The output additionally requires
`last_played_at < cutoff - recent_days` and records elapsed days since last play.
It exports all eligible user/item pairs to `candidates.parquet` with their counts,
cutoff, threshold and window. An empty result is valid. 360K is rejected because
aggregate artist counts cannot establish dormancy.

This is eligibility extraction only. It does not yet calculate contextual
nostalgia scores, infer emotional nostalgia, or blend Stream A and Stream B.

## Verification and next work

```powershell
uv run pytest -q
uv run python scripts/verify_streams.py
go test ./...
go vet ./...
```

`docs/streams-verified-state.json` is the portable milestone report; full local
artifacts and reports remain ignored by Git. Synthetic tags appear only in tests.
Next: obtain a temporally appropriate tag corpus, establish chronological holdout
evaluation and compare tag metrics/weights against ALS and popularity baselines.
Training loss reduction alone must not be presented as a measured relevance gain.

Implementation reference: https://benfred.github.io/implicit/api/models/cpu/als.html
