# Last.fm MSVD - data foundation

Repository: https://github.com/hrishike-sh/tag-recommendation

The repository includes source code, tests, reproducibility instructions, the
portable verification summary and `output/pdf/LastFM_Team_Handover.pdf`. Original
research/reference PDFs, raw datasets, generated matrices, local environments and
temporary files stay local. Acquire and rebuild the data with the commands below.
The handover records the September 15 milestone; its note that no remote existed
describes that milestone, before this repository was created.

Python owns acquisition, DuckDB ingestion, sparse training matrices, weighted
implicit ALS, Stream A discovery/tag reranking, and Stream B dormancy extraction.
Go provides the starting point for a later recommendation service and shares the
confidence contract. Acoustic features, contextual nostalgia scoring, stream
blending and a serving API are not implemented yet.

September 22 update: see [Stream A/B implementation](docs/streams.md). The 1K ALS
model is trained locally. TF-IDF and weighted overlap are implemented and tested;
real tag scoring awaits a tag corpus. The September 15 handover is historical.

## Quick start

Run from this repository root. Requires Python 3.12+, uv, curl and Go 1.24+.

```powershell
uv sync --frozen --extra dev
uv run lastfm acquire --dataset 1k
uv run lastfm acquire --dataset 360k
uv run lastfm ingest --dataset 1k
uv run lastfm ingest --dataset 360k
uv run lastfm matrix --dataset 1k --cutoff 2009-05-01T00:00:00Z --npz
uv run lastfm matrix --dataset 360k --npz
uv run lastfm inspect
uv run lastfm train --dataset 1k --factors 32 --iterations 10
uv run lastfm recommend --user-id user_000001 --k 10
uv run lastfm dormant --dataset 1k --min-historical-plays 5
uv run pytest -q
go test ./...
go run ./cmd/recommender -count 9 -kappa 40
```

In the prepared local workspace, `.venv/Scripts/python.exe -m lastfm.cli` is
equivalent to `uv run lastfm`. On Linux/macOS, use `uv run` or `.venv/bin/python`.
Use `lastfm --root /absolute/project/path ...` to operate from another directory.
The example cutoff is an initial training snapshot, not a tuned research decision.
Reserve later 1K events for validation/test; do not train on them.

## Dataset distinction

| Dataset | Unit | Time | Use |
| --- | --- | --- | --- |
| 1K | User-track listening event | UTC timestamp | Temporal training, future nostalgia experiments |
| 360K | User-artist total plays | No listening timestamp | Separate artist-level collaborative baseline |

These datasets are not concatenated: item granularity and user identifiers differ.
360K profile signup dates are not play dates. Profiles and original READMEs are
acquired and retained; demographic profile tables are not used by the model.
Neither archive contains an acoustic feature matrix or a usable tag corpus.

Original data is from Oscar Celma / UPF Music Technology Group and Last.fm.
The bundled READMEs restrict these datasets to non-commercial use. Data, virtual
environments, derived matrices and temporary files are excluded from Git.

Sources:
- https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
- https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz
- https://www.upf.edu/web/mtg/lastfm360k
- https://duckdb.org/docs/stable/data/partitioning/partitioned_writes.html

## Repository map

```text
python/lastfm/       ingestion, matrices, factorization, tags, discovery, dormancy, CLI
cmd/recommender/    Go confidence CLI; future serving entry point
internal/preference/ shared Go confidence formula and tests
tests/              Python integration and boundary tests
docs/               data contract, operations and handover notes
scripts/            verification and handover PDF builder
data/raw/           source archives, TSVs, READMEs, SHA-256 manifests (ignored)
data/lake/          immutable ingestion snapshots (ignored)
data/lastfm.duckdb   persistent views over Parquet (ignored)
artifacts/matrices/ sparse edges, mappings, metadata, optional NPZ (ignored)
artifacts/models/   trained factors, mappings, counts and model manifests (ignored)
artifacts/tags/     imported tag TF-IDF, relevance and coverage reports (ignored)
artifacts/dormancy/ eligible dormant user-track pairs (ignored)
artifacts/reports/  latest ingestion and verification results (ignored)
output/pdf/         teammate handover PDF
```

## DuckDB usage

```python
import duckdb
db = duckdb.connect('data/lastfm.duckdb', read_only=True)
db.execute("SET TimeZone='UTC'")
print(db.sql('SELECT * FROM recsys.events_1k LIMIT 5'))
print(db.sql('SELECT * FROM recsys.edges_1k LIMIT 5'))
```

Use explicit `year` and `month` filters together with `played_at` predicates for
Hive partition pruning. The catalog stores absolute Parquet paths: after moving
the repository, rebuild ingestion and matrices (or recreate views to the new paths).
Do not copy the DuckDB file by itself and expect the external data to travel with it.

## Scope and continuation

Read [the data contract](docs/data-contract.md) before writing a trainer and
[operations](docs/operations.md) before running the full pipeline. The initial
implementation is a local research data foundation, not a production recommender.
Next: run chronological evaluation, acquire a tag corpus, then add contextual
nostalgia scoring and stream blending. Current data/model verification does not
establish recommendation accuracy.

After full ingestion and matrix generation, run `uv run python scripts/verify.py`
to refresh the full-data audit and portable `docs/verified-state.json`. Rebuild the
handover with `uv run --extra docs python scripts/build_handover.py` after a passing
audit. The PDF goes to `output/pdf/LastFM_Team_Handover.pdf`.
