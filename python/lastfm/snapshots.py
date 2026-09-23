"""Read immutable matrix snapshots instead of mutable catalog views."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import duckdb
import numpy as np
from scipy.sparse import coo_matrix

from .acquire import sha256


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def dump(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def resolve_matrix(root: Path, dataset: str, matrix_dir: Path | None = None):
    if dataset not in {"1k", "360k"}:
        raise ValueError("Unknown dataset")
    if matrix_dir is None:
        report = json.loads((root / f"artifacts/reports/matrix-{dataset}.json").read_text())
        matrix_dir = Path(report["path"])
    matrix_dir = matrix_dir.resolve()
    metadata = json.loads((matrix_dir / "matrix.json").read_text())
    if metadata["dataset"] != dataset:
        raise ValueError("Dataset does not match matrix snapshot")
    return matrix_dir, metadata


def fingerprint(matrix_dir: Path) -> str:
    # Bind models/tags to the exact counts, cutoff and index mappings.
    value = {name: sha256(matrix_dir / name) for name in
             ("matrix.json", "users.parquet", "items.parquet", "edges.parquet")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_db():
    db = duckdb.connect()
    db.execute("SET TimeZone='UTC'")
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    return db


def load_counts(matrix_dir: Path, metadata: dict):
    # Always build from authoritative edges, not possibly stale optional NPZ files.
    with read_db() as db:
        data = db.execute("SELECT user_index,item_index,play_count FROM read_parquet(?)",
                          [str(matrix_dir / "edges.parquet")]).fetchnumpy()
    if not np.isfinite(data["play_count"]).all() or (data["play_count"] <= 0).any():
        raise ValueError("Counts must be positive and finite")
    counts = coo_matrix((data["play_count"], (data["user_index"], data["item_index"])),
                        shape=tuple(metadata["shape"])).tocsr()
    if counts.nnz != metadata["observed_pairs"]:
        raise ValueError("Edge count disagrees with matrix metadata or duplicate indices exist")
    return counts
