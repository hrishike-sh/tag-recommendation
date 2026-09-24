"""Versioned Parquet snapshots queried through a persistent DuckDB catalog."""
import json
from pathlib import Path
import uuid
import duckdb
from .acquire import sha256


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def connect(root: Path):
    (root / "data").mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(root / "data/lastfm.duckdb"))
    db.execute("SET TimeZone='UTC'")
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=4")
    db.execute("SET preserve_insertion_order=false")
    db.execute("CREATE SCHEMA IF NOT EXISTS recsys")
    return db


def ingest(root: Path, source: Path, dataset: str) -> dict:
    if dataset not in {"1k", "360k"}:
        raise ValueError("dataset must be 1k or 360k")
    digest = sha256(source)
    # A new directory per successful attempt avoids overwriting active Parquet readers.
    out = root / "data/lake" / dataset / (digest[:12] + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    with connect(root) as db:
        columns = (["user_id", "timestamp_raw", "artist_mbid", "artist_name", "track_mbid", "track_name"]
                   if dataset == "1k" else ["user_id", "artist_mbid", "artist_name", "plays_raw"])
        schema = "{" + ",".join(f"'{c}':'VARCHAR'" for c in columns) + "}"
        db.execute(f"""CREATE TEMP TABLE raw AS SELECT * FROM read_csv(
            {literal(source.resolve().as_posix())}, delim='\t', header=false, columns={schema},
            quote='', escape='', auto_detect=false, strict_mode=true,
            ignore_errors=true, store_rejects=true, rejects_limit=0)""")
        raw_rows = db.execute("SELECT count(*) FROM raw").fetchone()[0]
        parse_errors = db.execute("SELECT count(DISTINCT (scan_id,file_id,line)) FROM reject_errors").fetchone()[0]
        db.execute(f"COPY reject_errors TO {literal((out / 'parse_rejects.parquet').as_posix())} (FORMAT PARQUET)")
        # MBIDs are preferred. Name fallback is namespaced and artist-qualified for tracks.
        artist = "CASE WHEN nullif(trim(artist_mbid),'') IS NOT NULL THEN 'mbid:' || lower(trim(artist_mbid)) WHEN nullif(trim(artist_name),'') IS NOT NULL THEN 'name:' || sha256(lower(trim(artist_name))) END"
        track = f"CASE WHEN nullif(trim(track_mbid),'') IS NOT NULL THEN 'mbid:' || lower(trim(track_mbid)) WHEN ({artist}) IS NOT NULL AND nullif(trim(track_name),'') IS NOT NULL THEN 'name:' || sha256(to_json([({artist}),lower(trim(track_name))])) END"
        if dataset == "1k":
            db.execute(f"""CREATE TEMP TABLE normalized AS SELECT *,
                try_cast(timestamp_raw AS TIMESTAMPTZ) AS played_at,
                {artist} AS artist_id, {track} AS item_id FROM raw""")
            # Last.fm launched in 2002; this snapshot ends in June 2009.
            valid = """nullif(trim(user_id),'') IS NOT NULL AND item_id IS NOT NULL
                AND played_at >= TIMESTAMPTZ '2002-01-01 00:00:00+00'
                AND played_at < TIMESTAMPTZ '2009-07-01 00:00:00+00'"""
            db.execute(f"""CREATE TEMP TABLE clean AS SELECT DISTINCT trim(user_id) AS user_id,
                played_at, artist_id, item_id FROM normalized WHERE {valid}""")
            valid_rows = db.execute(f"SELECT count(*) FROM normalized WHERE {valid}").fetchone()[0]
            clean_rows = db.execute("SELECT count(*) FROM clean").fetchone()[0]
            db.execute(f"""COPY (SELECT *, year(played_at) AS year, month(played_at) AS month
                FROM clean ORDER BY played_at,user_id,item_id) TO {literal((out / 'events').as_posix())}
                (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(year,month))""")
            db.execute(f"""COPY (SELECT DISTINCT item_id, artist_id, artist_name, track_name,
                artist_mbid,track_mbid FROM normalized WHERE {valid})
                TO {literal((out / 'track_metadata.parquet').as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)""")
            view = "events_1k"
            glob = (out / "events/*/*/*.parquet").resolve().as_posix()
            extra = {"duplicates_removed": valid_rows - clean_rows,
                     "timestamp_min": str(db.execute("SELECT min(played_at) FROM clean").fetchone()[0]),
                     "timestamp_max": str(db.execute("SELECT max(played_at) FROM clean").fetchone()[0]),
                     "partitions": len(list((out / "events").glob("year=*/month=*"))),
                     "timestamp_policy": "UTC; [2002-01-01,2009-07-01); snapshot-specific outlier quarantine"}
        else:
            db.execute(f"""CREATE TEMP TABLE normalized AS SELECT *, {artist} AS item_id,
                try_cast(plays_raw AS BIGINT) AS play_count FROM raw""")
            valid = "regexp_full_match(trim(user_id),'[0-9a-f]{40}') AND item_id IS NOT NULL AND play_count > 0 AND regexp_full_match(trim(plays_raw),'[0-9]+')"
            valid_rows = db.execute(f"SELECT count(*) FROM normalized WHERE {valid}").fetchone()[0]
            db.execute(f"""CREATE TEMP TABLE clean AS SELECT trim(user_id) AS user_id, item_id,
                sum(play_count)::BIGINT AS play_count FROM normalized WHERE {valid} GROUP BY 1,2""")
            clean_rows = db.execute("SELECT count(*) FROM clean").fetchone()[0]
            db.execute(f"COPY clean TO {literal((out / 'artist_plays.parquet').as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
            db.execute(f"""COPY (SELECT DISTINCT item_id,artist_mbid,artist_name FROM normalized WHERE {valid})
                TO {literal((out / 'artist_metadata.parquet').as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)""")
            view = "artist_plays_360k"
            glob = (out / "artist_plays.parquet").resolve().as_posix()
            extra = {"merged_user_artist_rows": valid_rows - clean_rows,
                     "timestamp_policy": "No listening timestamps exist; no fabricated temporal partitions"}
        db.execute(f"""COPY (SELECT * FROM normalized WHERE NOT coalesce(({valid}),false))
            TO {literal((out / 'validation_rejects.parquet').as_posix())} (FORMAT PARQUET)""")
        report = {"dataset": dataset, "source": str(source.resolve()), "source_sha256": digest,
                  "raw_parsed_rows": raw_rows, "parse_rejected_rows": parse_errors,
                  "validation_rejected_rows": raw_rows - valid_rows, "clean_rows": clean_rows,
                  "users": db.execute("SELECT count(DISTINCT user_id) FROM clean").fetchone()[0],
                  "items": db.execute("SELECT count(DISTINCT item_id) FROM clean").fetchone()[0],
                  "snapshot": str(out.resolve()), **extra}
        if clean_rows == 0:
            raise ValueError("No valid rows; active catalog has not been changed")
        (out / "ingestion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        db.execute(f"CREATE OR REPLACE VIEW recsys.{view} AS SELECT * FROM read_parquet({literal(glob)}, hive_partitioning=true)")
        reports = root / "artifacts/reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"ingest-{dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
