"""Leakage-free chronological evaluation harness for Last.fm 1K."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD
from .baseline import PopularityRecommender
from .ingest import connect, literal
from .metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k


def split_events(
    root: Path,
    dataset: str = "1k",
    train_cutoff: str = "2009-05-01T00:00:00Z",
    test_end: Optional[str] = None,
    kappa: float = 40.0,
    min_train_interactions: int = 1,
) -> Dict[str, Any]:
    """Extract chronological train and test datasets from DuckDB views without future leakage.

    Parameters
    ----------
    root : Path
        Project root directory.
    dataset : str
        Dataset identifier (must be "1k").
    train_cutoff : str
        Strict exclusive UTC cutoff for training interactions.
    test_end : str or None
        Optional upper bound for test period.
    kappa : float
        Confidence scaling parameter.
    min_train_interactions : int
        Minimum training interactions for user eligibility.
    """
    if dataset != "1k":
        raise ValueError("Chronological evaluation requires timestamped events (1k dataset)")

    dt_train = datetime.fromisoformat(train_cutoff.replace("Z", "+00:00"))
    if dt_train.tzinfo is None:
        raise ValueError("train_cutoff must include a timezone, e.g. 2009-05-01T00:00:00Z")
    train_cutoff_iso = dt_train.astimezone(timezone.utc).isoformat()

    with connect(root) as db:
        end_train_lit = f"TIMESTAMPTZ {literal(train_cutoff_iso)}"

        # 1. Training counts (strictly before train_cutoff)
        db.execute(f"""
            CREATE TEMP TABLE train_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE (year < year({end_train_lit}) OR (year=year({end_train_lit}) AND month<=month({end_train_lit})))
              AND played_at < {end_train_lit}
            GROUP BY 1, 2
        """)

        # 2. Canonical training users and items mappings (0-based, lexicographical)
        db.execute("""
            CREATE TEMP TABLE eval_users AS
            SELECT user_id, (row_number() OVER (ORDER BY user_id)-1)::BIGINT AS user_index
            FROM (SELECT user_id, sum(play_count) AS total_plays FROM train_counts GROUP BY 1 HAVING sum(play_count) >= 1)
        """)

        db.execute("""
            CREATE TEMP TABLE eval_items AS
            SELECT item_id, (row_number() OVER (ORDER BY item_id)-1)::BIGINT AS item_index
            FROM (SELECT DISTINCT item_id FROM train_counts)
        """)

        # 3. Training edges
        db.execute(f"""
            CREATE TEMP TABLE train_edges AS
            SELECT u.user_index, i.item_index, tc.user_id, tc.item_id, tc.play_count,
                   1::UTINYINT AS preference,
                   {kappa} * ln(1.0 + tc.play_count) AS confidence_delta
            FROM train_counts tc
            JOIN eval_users u USING(user_id)
            JOIN eval_items i USING(item_id)
        """)

        # 4. Test events (strictly >= train_cutoff)
        test_where = f"played_at >= {end_train_lit}"
        if test_end:
            dt_end = datetime.fromisoformat(test_end.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            test_where += f" AND played_at < TIMESTAMPTZ {literal(dt_end)}"

        db.execute(f"""
            CREATE TEMP TABLE test_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE {test_where}
            GROUP BY 1, 2
        """)

        db.execute("""
            CREATE TEMP TABLE test_edges AS
            SELECT u.user_index, i.item_index, tc.user_id, tc.item_id, tc.play_count
            FROM test_counts tc
            JOIN eval_users u USING(user_id)
            JOIN eval_items i USING(item_id)
        """)

        # User eligibility stats
        all_train_users = db.execute("SELECT count(DISTINCT user_id) FROM train_counts").fetchone()[0]
        all_test_users = db.execute("SELECT count(DISTINCT user_id) FROM test_counts").fetchone()[0]
        evaluable_users_count = db.execute("SELECT count(DISTINCT user_index) FROM test_edges").fetchone()[0]

        train_data = db.execute("SELECT user_index, item_index, play_count, confidence_delta FROM train_edges").fetchnumpy()
        test_data = db.execute("SELECT user_index, item_index, play_count FROM test_edges").fetchnumpy()

        n_users = db.execute("SELECT count(*) FROM eval_users").fetchone()[0]
        n_items = db.execute("SELECT count(*) FROM eval_items").fetchone()[0]

    # Build matrices
    train_shape = (n_users, n_items)
    
    if len(train_data["user_index"]) > 0:
        indices = (train_data["user_index"], train_data["item_index"])
        P_train = sp.csr_matrix(
            (np.ones(len(train_data["user_index"]), dtype=np.float32), indices),
            shape=train_shape,
        )
        D_train = sp.csr_matrix(
            (train_data["confidence_delta"].astype(np.float32), indices),
            shape=train_shape,
        )
        Counts_train = sp.csr_matrix(
            (train_data["play_count"].astype(np.float32), indices),
            shape=train_shape,
        )
    else:
        P_train = sp.csr_matrix(train_shape, dtype=np.float32)
        D_train = sp.csr_matrix(train_shape, dtype=np.float32)
        Counts_train = sp.csr_matrix(train_shape, dtype=np.float32)

    # Group test positives by user_index
    test_positives_by_user: Dict[int, Set[int]] = {}
    for u_idx, i_idx in zip(test_data["user_index"], test_data["item_index"]):
        u = int(u_idx)
        i = int(i_idx)
        if u not in test_positives_by_user:
            test_positives_by_user[u] = set()
        test_positives_by_user[u].add(i)

    eligibility = {
        "all_dataset_users_in_training": all_train_users,
        "all_dataset_users_in_test": all_test_users,
        "evaluable_users_count": len(test_positives_by_user),
        "excluded_no_test_positives": n_users - len(test_positives_by_user),
        "min_train_interactions": min_train_interactions,
    }

    return {
        "P_train": P_train,
        "D_train": D_train,
        "Counts_train": Counts_train,
        "test_positives_by_user": test_positives_by_user,
        "n_users": n_users,
        "n_items": n_items,
        "train_cutoff": train_cutoff_iso,
        "test_end": test_end,
        "eligibility": eligibility,
    }


def evaluate_model(
    model: Any,
    P_train: sp.csr_matrix,
    test_positives_by_user: Dict[int, Set[int]],
    k_list: Sequence[int] = (5, 10, 20),
    exclude_seen: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate a trained model or baseline on all evaluable users."""
    evaluable_users = sorted(test_positives_by_user.keys())
    per_user_records: List[Dict[str, Any]] = []

    metrics_accum: Dict[str, float] = {}
    for k in k_list:
        metrics_accum[f"Recall@{k}"] = 0.0
        metrics_accum[f"Precision@{k}"] = 0.0
        metrics_accum[f"NDCG@{k}"] = 0.0
        metrics_accum[f"MAP@{k}"] = 0.0

    max_k = max(k_list)

    for u in evaluable_users:
        test_pos = test_positives_by_user[u]
        
        # Determine seen items
        start, end = P_train.indptr[u], P_train.indptr[u + 1]
        seen_items = set(P_train.indices[start:end]) if exclude_seen else set()

        recs = model.recommend(u, k=max_k, exclude_seen=exclude_seen, seen_items=seen_items)
        rec_item_indices = [item_idx for item_idx, _ in recs]

        user_metrics: Dict[str, Any] = {"user_index": u, "test_positives_count": len(test_pos)}

        for k in k_list:
            r = recall_at_k(rec_item_indices, test_pos, k)
            p = precision_at_k(rec_item_indices, test_pos, k)
            n = ndcg_at_k(rec_item_indices, test_pos, k)
            ap = average_precision_at_k(rec_item_indices, test_pos, k)

            metrics_accum[f"Recall@{k}"] += r
            metrics_accum[f"Precision@{k}"] += p
            metrics_accum[f"NDCG@{k}"] += n
            metrics_accum[f"MAP@{k}"] += ap

            user_metrics[f"Recall@{k}"] = r
            user_metrics[f"Precision@{k}"] = p
            user_metrics[f"NDCG@{k}"] = n
            user_metrics[f"MAP@{k}"] = ap

        per_user_records.append(user_metrics)

    n_eval = len(evaluable_users)
    macro_metrics: Dict[str, float] = {}
    if n_eval > 0:
        for metric_name, total_val in metrics_accum.items():
            macro_metrics[metric_name] = total_val / float(n_eval)

    return macro_metrics, per_user_records


def run_experiment(
    root: Path,
    model_type: str = "als",
    dataset: str = "1k",
    train_cutoff: str = "2009-05-01T00:00:00Z",
    test_end: Optional[str] = None,
    factors: int = 32,
    regularization: float = 0.05,
    iterations: int = 15,
    seed: int = 42,
    kappa: float = 40.0,
    k_list: Sequence[int] = (5, 10, 20),
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run an end-to-end evaluation run and save machine-readable results."""
    run_id = f"{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_dir = (output_dir or (root / "results" / run_id)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and splitting dataset '{dataset}' (cutoff: {train_cutoff})...")
    split = split_events(root, dataset=dataset, train_cutoff=train_cutoff, test_end=test_end, kappa=kappa)

    P_train = split["P_train"]
    D_train = split["D_train"]
    Counts_train = split["Counts_train"]
    test_pos = split["test_positives_by_user"]

    print(f"Split completed: {split['n_users']} users, {split['n_items']} items. Evaluable users: {len(test_pos)}")

    train_start = time.perf_counter()
    if model_type.lower() in {"als", "msvd", "implicit_msvd"}:
        model = ImplicitMSVD(factors=factors, regularization=regularization, iterations=iterations, seed=seed)
        model.fit(P_train, D_train, compute_loss_history=False, verbose=False)
    elif model_type.lower() in {"pop", "popularity"}:
        model = PopularityRecommender()
        model.fit(Counts_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    train_time = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    macro_metrics, per_user_metrics = evaluate_model(model, P_train, test_pos, k_list=k_list, exclude_seen=True)
    eval_time = time.perf_counter() - eval_start

    config = {
        "run_id": run_id,
        "model_type": model_type,
        "dataset": dataset,
        "train_cutoff": split["train_cutoff"],
        "test_end": split["test_end"],
        "factors": factors if "als" in model_type else None,
        "regularization": regularization if "als" in model_type else None,
        "iterations": iterations if "als" in model_type else None,
        "seed": seed,
        "kappa": kappa,
        "k_list": list(k_list),
        "n_users": split["n_users"],
        "n_items": split["n_items"],
        "evaluable_users": len(test_pos),
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "user_eligibility.json").write_text(json.dumps(split["eligibility"], indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(macro_metrics, indent=2), encoding="utf-8")
    (out_dir / "per_user_metrics.json").write_text(json.dumps(per_user_metrics, indent=2), encoding="utf-8")

    return {
        "config": config,
        "macro_metrics": macro_metrics,
        "eligibility": split["eligibility"],
        "output_dir": str(out_dir),
    }
