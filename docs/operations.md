# Local operation and teammate handoff

## Reproduce

Use the commands in README.md from the repo root. `uv.lock` fixes the Python
dependency resolution. The Go module currently uses only the standard library.
`go run ./cmd/recommender` is a formula smoke test, not a trained recommender.

Acquisition uses curl with HTTPS certificate verification, retry and resume.
Archives are extracted with Python's data filter; archive paths must stay below
`data/raw`, and bundled scripts are not executed. TSVs, README/license material
and original archives are retained. Manifests record SHA-256 and byte lengths.
These are locally computed integrity fingerprints, not signed publisher proofs.
Subsequent acquisition checks the stored hashes before returning cached data.

## Storage and concurrency

Allow at least 15 GB free for archives, extracted files, Parquet, matrix exports
and DuckDB spill space. Each ingestion uses four threads and a 2 GB DuckDB memory
limit. Python sparse export consumes additional memory outside this limit.
Run catalog writers sequentially; DuckDB is not a multi-process writer service.
Readers should open the catalog read-only while no writer owns its process lock.

Failure before view publication leaves the previous active snapshot intact.
Partial snapshot folders may remain, and old snapshots are not automatically
deleted. Inspect `ingestion.json`, `matrix.json` and catalog view definitions before
manually removing any snapshot. Never delete the active Parquet directory.

The timestamp policy is specific to the supplied 1K release. Monthly partitioning
is a balance between time pruning and small-file overhead. Prefer query predicates
on `year, month` plus the exact UTC timestamp condition.

## Useful checks

```sql
SELECT count(*), min(played_at), max(played_at) FROM recsys.events_1k;
SELECT count(*), sum(play_count) FROM recsys.artist_plays_360k;
SELECT count(*) FROM recsys.edges_1k
WHERE active_plays + historical_plays != play_count;
EXPLAIN SELECT count(*) FROM recsys.events_1k
WHERE year=2009 AND month=4;
```

Raw profiles are downloaded but not ingested or exposed by the Go CLI. Tag data,
audio features, trained factors, recommendation endpoints and an evaluation suite
remain future work. Do not report recommendation accuracy from this ingestion.

## Suggested work allocation

| Workstream | First deliverable | Dependency |
| --- | --- | --- |
| Data / metadata | Tag-source coverage report and identity mapping audit | Canonical item mappings |
| Modeling | Implicit ALS baseline consuming the confidence contract | Sparse matrix snapshot |
| Evaluation | Validation/test chronology, Recall/NDCG and cold-start report | Training cutoff agreement |
| Nostalgia | Dormant candidate retrieval using historical/active counts | Evaluation definition |
| Go serving | Load versioned model artifacts and expose recommendation API | Trained factors and mapping contract |

The team should agree on kappa tuning, cutoff dates, dormancy thresholds, tag
fallbacks and novelty/nostalgia score calibration before claiming improvements.
