"""Stream A latent core: confidence-weighted implicit ALS, not SVD of C*P."""
from datetime import datetime, timezone
import importlib.metadata
import math
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits

from .ingest import literal
from .snapshots import dump, fingerprint, load_counts, read_db, resolve_matrix


def als_confidence(counts, kappa):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    confidence = counts.astype(np.float32, copy=True).tocsr()
    if (confidence.data <= 0).any() or not np.isfinite(confidence.data).all():
        raise ValueError("Observed counts must be positive and finite")
    confidence.data = 1 + kappa * np.log1p(confidence.data)
    if not np.isfinite(confidence.data).all():
        raise ValueError("Confidence overflow")
    # implicit expects FULL observed confidence, not C-1. Unstored pairs imply C=1.
    return confidence


def fit_als(counts, kappa=40, factors=32, regularization=0.1, iterations=10,
            seed=42, threads=4, callback=None, exact=False):
    from implicit.cpu.als import AlternatingLeastSquares
    if factors < 1 or iterations < 1 or threads < 1:
        raise ValueError("factors, iterations and threads must be positive")
    if regularization <= 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    if min(counts.shape) == 0 or counts.nnz == 0:
        raise ValueError("Cannot train on an empty matrix")
    confidence = als_confidence(counts, kappa)
    with threadpool_limits(limits=1, user_api="blas"):
        model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, alpha=1.0,
            iterations=iterations, random_state=seed, num_threads=threads,
            calculate_training_loss=True, use_cg=not exact)
        model.fit(confidence, show_progress=False, callback=callback)
    if not np.isfinite(model.user_factors).all() or not np.isfinite(model.item_factors).all():
        raise ValueError("Non-finite trained factors")
    return model


def copy_item_metadata(matrix_dir, matrix_meta, out):
    source = matrix_meta.get("source") or {}
    if "snapshot" not in source:
        return False
    filename = "track_metadata.parquet" if matrix_meta["dataset"] == "1k" else "artist_metadata.parquet"
    source_file = Path(source["snapshot"]) / filename
    if not source_file.exists():
        return False
    if matrix_meta["dataset"] == "1k":
        fields = "item_id,artist_id,artist_name,track_name"
        ordering = "artist_id NULLS LAST,artist_name NULLS LAST,track_name NULLS LAST"
    else:
        fields = "item_id,item_id AS artist_id,artist_name,NULL::VARCHAR AS track_name"
        ordering = "artist_name NULLS LAST"
    with read_db() as db:
        db.execute(f"""COPY (WITH names AS (
            SELECT {fields}, row_number() OVER (PARTITION BY item_id ORDER BY {ordering}) AS rn
            FROM read_parquet({literal(source_file)}))
            SELECT m.item_index,m.item_id,n.artist_id,n.artist_name,n.track_name
            FROM read_parquet({literal(matrix_dir / 'items.parquet')}) m
            LEFT JOIN names n ON m.item_id=n.item_id AND n.rn=1)
            TO {literal(out / 'item_metadata.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    return True


def train(root: Path, dataset="1k", matrix_dir=None, factors=32, regularization=0.1,
          iterations=10, seed=42, threads=4):
    matrix_dir, matrix_meta = resolve_matrix(root, dataset, matrix_dir)
    matrix_id = fingerprint(matrix_dir)
    counts = load_counts(matrix_dir, matrix_meta)
    history = []
    def progress(epoch, elapsed, loss):
        entry = {"epoch": epoch + 1, "seconds": float(elapsed), "library_training_loss": float(loss)}
        history.append(entry)
        print(f"ALS epoch {epoch+1}/{iterations}: loss={loss:.7f}, {elapsed:.2f}s", file=sys.stderr, flush=True)
    started = time.perf_counter()
    model = fit_als(counts, matrix_meta["kappa"], factors, regularization,
                    iterations, seed, threads, progress)
    out = root / "artifacts/models" / dataset / uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    np.save(out / "user_factors.npy", model.user_factors, allow_pickle=False)
    np.save(out / "item_factors.npy", model.item_factors, allow_pickle=False)
    save_npz(out / "counts.npz", counts)
    for filename in ("users.parquet", "items.parquet", "matrix.json"):
        shutil.copyfile(matrix_dir / filename, out / filename)
    content_available = copy_item_metadata(matrix_dir, matrix_meta, out)
    metadata = {
        "schema_version": 1, "algorithm": "confidence-weighted implicit ALS + optional tag/content fusion",
        "msvd_scope": "Implicit adaptation of the tag-M SVD pipeline; original user/genre regularizers are unspecified and not reproduced",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
        "matrix_fingerprint": matrix_id, "matrix_path": str(matrix_dir),
        "shape": list(counts.shape), "observed_pairs": counts.nnz,
        "cutoff": matrix_meta["exclusive_cutoff_utc"], "kappa": matrix_meta["kappa"],
        "factors": factors, "regularization": regularization, "iterations": iterations,
        "seed": seed, "threads": threads, "solver": "CPU ALS with conjugate gradient",
        "implicit_version": importlib.metadata.version("implicit"),
        "content_features": "one-hot primary artist identity" if content_available else None,
        "training_seconds": time.perf_counter()-started, "training_history": history,
        "path": str(out.resolve()), "quality_evaluation": "not performed"}
    dump(out / "model.json", metadata)
    dump(root / f"artifacts/reports/model-{dataset}.json", metadata)
    return metadata
