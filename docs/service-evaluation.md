# Shared service evaluation and retrieval

The CLI, browser jobs, and evaluation now use `RecommendationService`. Both streams share the fitted factors, pre-cutoff tag matrix, item mapping, and explanation schema. The legacy `--mode global` path remains available for the earlier three-signal experiments.

## Run the service

From the project directory in PowerShell:

```powershell
uv sync --extra dev --extra evaluation --extra ann
uv run --no-sync lastfm-service prepare
uv run --no-sync lastfm-web
```

Open http://127.0.0.1:8766/. Use **Recommendation studio** for mixes and **Evaluation lab** for quality or latency experiments. The first preparation fits the default 64-factor, 10-iteration ALS model. Later jobs load the saved model; jobs still use isolated worker processes, so bundle loading and history queries incur startup cost. This is a local research service, not a deployed multi-user API.

```powershell
uv run --no-sync lastfm-service recommend user_000001 --k 10 --output artifacts/reports/mix.json
uv run --no-sync lastfm-service recommend user_000001 --retrieval hnsw --candidates 500
uv run --no-sync lastfm-service evaluate --max-users 20 --k 10 --output artifacts/reports/service-quality.json
uv run --no-sync lastfm-service evaluate --max-users 0 --k 20 --output artifacts/reports/service-quality-all.json
uv run --no-sync lastfm-service benchmark --max-users 10 --k 10 --output artifacts/reports/service-latency.json
uv run --no-sync pytest -q
```

`--max-users 0` requests the full cohort at the CLI. The UI caps jobs at 100 listeners. Quality cohorts use a fixed seed of 42; latency cohorts use the first sorted training listeners. Full training and HNSW construction can take several minutes on the 1K catalog. `--factors` and `--iterations` go before the subcommand and select a distinct cached bundle. For example, `lastfm-service --factors 8 --iterations 1 prepare` creates a smoke-test model, not the default research model.

## Snapshot and service contract

Bundles live under ignored `artifacts/service/<key>/`. They contain NPZ factors, sparse preferences/confidence/tag matrices, JSON metadata, ordered user and item IDs, and the optional `hnsw.faiss` index. No pickle is loaded. The bundle key incorporates the ingestion report, tag file names/sizes/modification times, model settings, cutoff, and service schema version. Snapshot files must remain immutable; after changing scoring or feature semantics, bump the schema version in `service.py` before reusing caches. This is cache invalidation by local snapshot metadata, not a cryptographic audit of every file.

Training uses events strictly before 1 May 2009. Historical tag assignments are also strictly before this cutoff. History reads use the bundle's fixed Parquet snapshot. Stream A ranks unheard tracks using collaborative and tag affinity. Stream B retains eligible dormant tracks even though they have been heard before. Equation 9 and the dynamic alpha rule remain in `nostalgia.py`.

`recommend()` returns ordered items, alpha and its mode, normalized stream contributions, explanation tuples and their field names, nostalgia evidence, service key, retrieval mode, candidate count, and ranking latency. The existing `lastfm recommend` dual mode delegates to this service. A recommendation request does not use holdout outcomes to score candidates.

## Evaluation protocol

This implementation uses **scikit-learn**, the selected option for the requested RecBole/scikit-learn harness. It does not install or execute RecBole. The older evaluation module remains intact for earlier experiments.

The default holdout is `[2009-05-01, 2009-07-01)`. Users and items absent from training are excluded and their holdout pair counts are reported. Relevant targets are future plays of either unheard catalog tracks or tracks meeting the dormancy rule at the cutoff. Active previously heard tracks are not eligible recommendations and are excluded from the relevance denominator. Users with no eligible holdout targets are excluded and counted. No random interaction split or negative sampling is used.

| Metric | Definition | Missing cases |
| --- | --- | --- |
| NDCG@K | Binary relevance DCG divided by ideal DCG, with logarithmic rank discount; implemented with scikit-learn `dcg_score` | Null when no relevant targets; short lists receive zero gain for unfilled positions |
| Novelty@K | Mean `-log2((df_i + 1)/(N + 1))` over returned items; `df_i` is the number of training listeners for item i and N is training listeners | Null for an empty list; units are bits, not a percentage |
| Diversity@K | Mean pairwise cosine distance between recommendations with nonempty TF-IDF tag rows | Null with fewer than two tagged recommendations; tag coverage is reported separately |
| Nostalgia recall@K | Returned dormant holdout positives divided by all dormant holdout positives | Null if the listener has no dormant holdout positives; zero if targets exist but none are returned |

Dormant holdout positives are defined by history and future replay, not by whether Stream B assigned a positive score. Otherwise missing tags would artificially shrink the denominator and inflate recall. This is a replay proxy, not a measured emotional response. Metrics are macro-averaged over users with defined values; each metric reports its own user count. JSON includes per-user values, cohort settings, exclusions, and service key.

## FAISS HNSW prototype

HNSW indexes the collaborative item factors using maximum inner product. Vectors are not cosine-normalized because ALS uses their magnitudes. Search overfetches enough candidates to remove seen tracks, unions the collaborative shortlist with the strongest positive tag candidates, and then blends with **all** eligible dormant tracks. Stream B is never restricted to the ANN shortlist.

Exact retrieval remains the default. HNSW uses min-max normalization within the collaborative shortlist with a zero anchor; it can therefore change both candidate recall and score scaling relative to full-catalog normalization. The output labels this approximation. `--candidates` controls each candidate source; their union can be larger. M=32, efConstruction=100, and efSearch=128 are initial settings, not tuned optima.

The benchmark reports exact and HNSW p50/p95 latency, top-K overlap with exact ranking, and index build/load time separately. It warms the history and both retrieval paths first and alternates measurement order. Warm request timing includes candidate retrieval, scoring, blending, and explanations; it excludes process startup, model loading/training, history I/O, and first index construction. It is not HTTP response latency. Indexing does not eliminate all catalog-wide work: tag scoring, dormancy preparation, and blending still have costs. Measure before claiming a speed improvement.

## Demonstration and evidence

`tests/test_service.py` creates a small deterministic two-user fixture with both a replayed dormant track and a new holdout track. It checks persistence without retraining, future-data exclusion, hand-calculated metrics, explanations, HNSW exclusions, and index reload. The browser workflow is exercised against an isolated synthetic workspace under `tmp/`; its results must not be presented as Last.fm research performance.

The separate real-data smoke run uses eight factors and one ALS iteration on the full pre-cutoff catalog. Its quality report uses 20 seeded holdout listeners and the latency report uses three listeners with three repeats. These are pipeline checks, not final model comparisons. Raw data, bundles, indexes, and listener-level reports stay ignored. A compact aggregate validation summary is suitable for version control.

## References

- [scikit-learn DCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html)
- [scikit-learn cosine distance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)
- [FAISS HNSW API](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWFlat.html)
- [FAISS inner-product metric](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
