"""Sparse implicit preferences; missing edges have p=0 and confidence=1."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid
from .ingest import connect, literal


def preference_confidence(count: int, kappa: float = 40.0) -> tuple[int, float]:
    if count < 0 or not math.isfinite(kappa) or kappa < 0:
        raise ValueError("count and kappa must be nonnegative; kappa must be finite")
    return int(count > 0), 1.0 + kappa * math.log1p(count)


def matrix(root: Path, dataset: str, cutoff: str | None = None, kappa: float = 40,
           recent_days: int = 90, export_npz: bool = False) -> dict:
    preference_confidence(0, kappa)
    if recent_days <= 0:
        raise ValueError("recent_days must be positive")
    if dataset == "1k" and cutoff is None:
        raise ValueError("1k requires an explicit exclusive UTC cutoff to prevent leakage")
    if dataset == "360k" and cutoff is not None:
        raise ValueError("360k has no timestamps and cannot support a temporal cutoff")
    if dataset not in {"1k", "360k"}:
        raise ValueError("dataset must be 1k or 360k")
    if cutoff:
        dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("cutoff must include a timezone, e.g. 2009-05-01T00:00:00Z")
        cutoff = dt.astimezone(timezone.utc).isoformat()
    out = root / "artifacts/matrices" / dataset / uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    with connect(root) as db:
        if dataset == "1k":
            end = f"TIMESTAMPTZ {literal(cutoff)}"
            # Explicit partition predicates enable Hive file pruning.
            db.execute(f"""CREATE TEMP TABLE counts AS SELECT user_id,item_id,
                count(*)::BIGINT AS play_count, max(played_at) AS last_played_at,
                count(*) FILTER (WHERE played_at >= {end} - INTERVAL '{recent_days} days')::BIGINT AS active_plays,
                count(*) FILTER (WHERE played_at < {end} - INTERVAL '{recent_days} days')::BIGINT AS historical_plays
                FROM recsys.events_1k
                WHERE (year < year({end}) OR (year=year({end}) AND month<=month({end})))
                AND played_at < {end} GROUP BY 1,2""")
        else:
            db.execute("CREATE TEMP TABLE counts AS SELECT * FROM recsys.artist_plays_360k")
        for kind in ("user", "item"):
            db.execute(f"""CREATE TEMP TABLE {kind}s AS SELECT {kind}_id,
                (row_number() OVER (ORDER BY {kind}_id)-1)::BIGINT AS {kind}_index
                FROM (SELECT DISTINCT {kind}_id FROM counts)""")
        db.execute(f"""CREATE TEMP TABLE edges AS SELECT user_index,item_index,c.*,
            1::UTINYINT AS preference, {kappa} * ln(1.0+play_count) AS confidence_delta,
            1.0+{kappa} * ln(1.0+play_count) AS confidence
            FROM counts c JOIN users USING(user_id) JOIN items USING(item_id)""")
        for table in ("users", "items", "edges"):
            db.execute(f"COPY {table} TO {literal((out / (table+'.parquet')).as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        shape = [db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("users", "items")]
        nnz = db.execute("SELECT count(*) FROM edges").fetchone()[0]
        if nnz == 0:
            raise ValueError("No training interactions before cutoff")
        if export_npz:
            import numpy as np
            from scipy.sparse import coo_matrix, save_npz
            data = db.execute("SELECT user_index,item_index,play_count,confidence_delta FROM edges").fetchnumpy()
            indices = (data["user_index"], data["item_index"])
            for name, values in (("counts",data["play_count"]),
                                 ("preferences",np.ones(nnz,dtype=np.float32)),
                                 ("confidence_delta",data["confidence_delta"])):
                sparse = coo_matrix((values,indices),shape=tuple(shape)).tocsr()
                sparse.eliminate_zeros()
                save_npz(out / f"{name}.npz",sparse)
        source_report = root / "artifacts/reports" / f"ingest-{dataset}.json"
        metadata = {"dataset": dataset, "item_type": "track" if dataset == "1k" else "artist",
                    "shape": shape, "observed_pairs": nnz, "kappa": kappa,
                    "exclusive_cutoff_utc": cutoff, "recent_days": recent_days if dataset == "1k" else None,
                    "preference": "1[count > 0]", "confidence": "1 + kappa * ln(1 + count)",
                    "missing_pair": {"preference": 0, "confidence": 1, "confidence_delta": 0},
                    "source": json.loads(source_report.read_text()) if source_report.exists() else None,
                    "path": str(out.resolve()), "npz_exported": export_npz}
        (out / "matrix.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        db.execute("BEGIN TRANSACTION")
        try:
            for table in ("edges", "users", "items"):
                db.execute(f"CREATE OR REPLACE VIEW recsys.{table}_{dataset} AS SELECT * FROM read_parquet({literal((out / (table+'.parquet')).resolve().as_posix())})")
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        reports = root / "artifacts/reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"matrix-{dataset}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
