# Data and confidence contract

## Canonical ingestion

1K source fields: `user_id, timestamp_raw, artist_mbid, artist_name, track_mbid,
track_name`. Canonical events contain `user_id VARCHAR, played_at TIMESTAMPTZ,
artist_id VARCHAR, item_id VARCHAR, year BIGINT, month BIGINT`. UTC `year` and
`month` are physical Hive directory partitions, not native DuckDB table partitions.
Data is stored in ZSTD-compressed Parquet and exposed as `recsys.events_1k`.

360K source fields: `user_id, artist_mbid, artist_name, plays_raw`. Canonical
`recsys.artist_plays_360k` contains `user_id, item_id, play_count BIGINT`.
It is deliberately not partitioned by time.

IDs prefer lowercase, trimmed MusicBrainz IDs with an `mbid:` prefix. Missing
artist MBIDs fall back to `name:` + SHA-256(lowercase trimmed artist name).
Missing track MBIDs fall back to a SHA-256 of the JSON pair `[artist_id, track_name]`.
Fallbacks retain items rather than dropping everything without MBIDs, but they do
not resolve aliases or merge a name-only track with its MBID-bearing equivalent.
Metadata Parquet files retain source names and MBIDs; multiple aliases can exist.

## Cleaning and audit

- Malformed TSV records are quarantined in `parse_rejects.parquet`; they are not
  silently discarded. `parse_rejected_rows` counts distinct rejected source lines.
- Missing user IDs/item identities, invalid dates and nonpositive/invalid artist
  counts go to `validation_rejects.parquet`.
- 360K user IDs must match 40 lowercase hexadecimal characters. The original file
  includes date strings in the user-ID field; these are quarantined, not treated
  as users. Play counts must be positive integer strings, never rounded decimals.
- 1K accepts timestamps in `[2002-01-01, 2009-07-01)` UTC. This is an explicit policy
  for this historical release. The bundled README says May 2009, whereas actual
  timestamp coverage is measured in the ingestion report; do not assume the text
  describes the exact final event. Change bounds before ingesting another release.
- 1K collapses exact canonical `(user_id, played_at, artist_id, item_id)` duplicates.
  Multiple different timestamps still count as repeated listens. This policy may
  collapse indistinguishable simultaneous events because no source event ID exists.
- 360K sums repeated canonical user-artist rows. These are aggregate records,
  so repeat rows cannot be interpreted as timestamped individual plays.
- Every run produces a new immutable snapshot; successful publication replaces the
  active view rather than appending rows. Reruns do not double the active dataset.
- `ingestion.json` records SHA-256, row counts, rejected counts, cardinalities and
  time coverage. Prior snapshots remain available for audit.

## Preference matrix

For a user `u` and item `i`, define `n_ui` as the nonnegative play count.

```text
P_ui = 1 if n_ui > 0, otherwise 0
C_ui = 1 + kappa * ln(1 + n_ui)
D_ui = C_ui - 1 = kappa * ln(1 + n_ui)
```

`kappa=40` is the configurable initial default, not a tuned result. It must be
finite and nonnegative. The logarithm is natural. Do not confuse kappa with the
future discovery/nostalgia mixing weight alpha.

| Plays | Preference | Confidence at kappa=40 |
| ---: | ---: | ---: |
| 0 | 0 | 1.000000 |
| 1 | 1 | 28.725887 |
| 9 | 1 | 93.103404 |

For 1K, `n_ui` counts cleaned events strictly before the explicit timezone-aware
cutoff. Events at the cutoff are excluded. `active_plays` uses
`[cutoff - recent_days, cutoff)` and `historical_plays` uses the earlier interval.
Their sum equals `play_count`; `last_played_at` also uses training history only.
The default recent window is 90 days. Counts in the prepared run use all available
pre-cutoff history; there is no lower training bound beyond ingestion validity.

For 360K, `n_ui` is the provided artist total. A temporal cutoff is rejected because
the source has no listening timestamps. Never call its results temporal evaluation.

## Stored representation and training objective

Each matrix snapshot has deterministic zero-based, lexicographically sorted user
and item mappings (`users.parquet`, `items.parquet`), plus `edges.parquet` with
`user_index, item_index, user_id, item_id, play_count, preference, confidence_delta,
confidence` and the 1K temporal fields. Mappings are stable for the same snapshot,
but may change when the training population changes. Always ship them with a model.

Optional NPZ exports are CSR matrices: `counts.npz`, `preferences.npz`, and
`confidence_delta.npz`. Missing entries in these sparse files are zero.
**The actual confidence of a missing pair is ONE, not zero.**
Use `C = all-ones baseline + D` conceptually, without allocating a dense matrix.
The edges Parquet alone already defines the sparse matrix.

The intended implicit weighted least-squares objective is:

```text
min over X,Y: sum over all users and items
    C_ui * (P_ui - dot(X_u,Y_i))^2
    + lambda_reg * (sum ||X_u||^2 + sum ||Y_i||^2)
```

The sum includes unobserved pairs at confidence 1. It is not simply SVD of `C*P`,
and fitting only stored positive edges would optimize a different objective.
Check a chosen ALS library's sparse-confidence convention before passing these
exports. No solver or user/genre regularizers have been implemented in this step.

Only users/items observed before the cutoff get training indices. Unseen holdout
items/users must be reported as cold start in subsequent evaluation. No cross-user
identity join between 1K and 360K is defined.
