"""Execute full-scale Phase 5 baseline evaluation on canonical Last.fm 1K dataset."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc
import duckdb
import numpy as np

from lastfm.als import ImplicitMSVD
from lastfm.baseline import PopularityRecommender
from lastfm.evaluate import evaluate_model
from lastfm.ingest import connect
from lastfm.split import split_train_val_test


def main():
    root = Path.cwd()
    out_dir = root / "results/phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_dir = out_dir / "popularity"
    pop_dir.mkdir(parents=True, exist_ok=True)
    als_dir = out_dir / "als"
    als_dir.mkdir(parents=True, exist_ok=True)
    cold_start_dir = out_dir / "cold_start"
    cold_start_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("PHASE 5: FULL-SCALE CANONICAL LAST.FM 1K BASELINE EXECUTION")
    print("=================================================================")

    # 1. Verify Active DuckDB View
    print("\n--- 1. Verifying Canonical recsys.events_1k ---")
    with connect(root) as db:
        cnt, n_u, n_i, t_min, t_max = db.execute("""
            SELECT count(*), count(DISTINCT user_id), count(DISTINCT item_id),
                   min(played_at), max(played_at)
            FROM recsys.events_1k
        """).fetchone()

    dataset_summary = {
        "dataset": "1k",
        "clean_rows": cnt,
        "users": n_u,
        "items": n_i,
        "timestamp_min": str(t_min),
        "timestamp_max": str(t_max),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print(f"Total Events: {cnt:,}")
    print(f"Distinct Users: {n_u}")
    print(f"Distinct Tracks: {n_i:,}")
    print(f"Time Span: {t_min} to {t_max}")

    # 2. Chronological Split
    print("\n--- 2. Performing Canonical Chronological Split ---")
    t0_split = time.perf_counter()
    splits = split_train_val_test(
        root,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        test_end="2009-07-01T00:00:00Z",
        kappa=40.0,
    )
    split_time = time.perf_counter() - t0_split

    val_split = splits["validation_split"]
    test_split = splits["test_split"]

    P_train_test = test_split["P_train"]
    D_train_test = test_split["D_train"]
    Counts_train_test = test_split["Counts_train"]
    test_positives = test_split["test_positives"]

    # Cold start calculations
    with connect(root) as db:
        # Train item set (< 2009-05-01)
        train_items = set(r[0] for r in db.execute("""
            SELECT DISTINCT item_id FROM recsys.events_1k WHERE played_at < TIMESTAMPTZ '2009-05-01 00:00:00+00'
        """).fetchall())
        
        # Test item set (>= 2009-05-01)
        test_events = db.execute("""
            SELECT user_id, item_id FROM recsys.events_1k WHERE played_at >= TIMESTAMPTZ '2009-05-01 00:00:00+00'
        """).fetchall()

    total_test_interactions = len(test_events)
    test_events_in_train = sum(1 for _, item in test_events if item in train_items)
    test_events_future_only = total_test_interactions - test_events_in_train
    future_only_items = len(set(item for _, item in test_events if item not in train_items))
    cold_start_unreachable_pct = (test_events_future_only / float(total_test_interactions)) * 100.0 if total_test_interactions > 0 else 0.0

    cold_start_summary = {
        "total_test_interactions": total_test_interactions,
        "test_interactions_in_training_catalog": test_events_in_train,
        "test_interactions_future_only_items": test_events_future_only,
        "future_only_tracks_count": future_only_items,
        "cold_start_unreachable_percentage": cold_start_unreachable_pct,
    }
    (cold_start_dir / "summary.json").write_text(json.dumps(cold_start_summary, indent=2), encoding="utf-8")

    matrix_summary = {
        "train_shape": list(P_train_test.shape),
        "train_nnz": int(P_train_test.nnz),
        "train_density_pct": (P_train_test.nnz / float(P_train_test.shape[0] * P_train_test.shape[1])) * 100.0,
        "evaluable_test_users": len(test_positives),
        "split_construction_time_sec": split_time,
    }
    (out_dir / "matrix_summary.json").write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")

    print(f"Train Matrix Shape: {P_train_test.shape[0]} users x {P_train_test.shape[1]:,} items")
    print(f"Train NNZ: {P_train_test.nnz:,}")
    print(f"Evaluable Test Users: {len(test_positives)}")
    print(f"Cold-Start Test Interactions: {test_events_future_only:,} / {total_test_interactions:,} ({cold_start_unreachable_pct:.2f}% unreachable by pure ALS)")

    # 3. Full-Scale Popularity Baseline
    print("\n--- 3. Running Full-Scale Popularity Baseline ---")
    t0_pop = time.perf_counter()
    pop = PopularityRecommender().fit(Counts_train_test)
    pop_train_time = time.perf_counter() - t0_pop

    t0_pop_eval = time.perf_counter()
    pop_metrics, pop_user_metrics = evaluate_model(pop, P_train_test, test_positives, k_list=[5, 10, 20], exclude_seen=True)
    pop_eval_time = time.perf_counter() - t0_pop_eval

    (pop_dir / "metrics.json").write_text(json.dumps(pop_metrics, indent=2), encoding="utf-8")
    (pop_dir / "per_user_metrics.json").write_text(json.dumps(pop_user_metrics, indent=2), encoding="utf-8")

    print("Popularity Test Results:")
    for k in [5, 10, 20]:
        print(f"  Recall@{k:2d}: {pop_metrics[f'Recall@{k}']:.4f} | NDCG@{k:2d}: {pop_metrics[f'NDCG@{k}']:.4f} | Precision@{k:2d}: {pop_metrics[f'Precision@{k}']:.4f} | MAP@{k:2d}: {pop_metrics[f'MAP@{k}']:.4f}")

    # 4. Full-Scale Implicit MSVD
    print("\n--- 4. Running Full-Scale Implicit MSVD (factors=64, reg=0.05, iters=10) ---")
    tracemalloc.start()
    t0_als = time.perf_counter()
    als = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    als.fit(P_train_test, D_train_test, compute_loss_history=False, verbose=True)
    als_train_time = time.perf_counter() - t0_als
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_als_eval = time.perf_counter()
    als_metrics, als_user_metrics = evaluate_model(als, P_train_test, test_positives, k_list=[5, 10, 20], exclude_seen=True)
    als_eval_time = time.perf_counter() - t0_als_eval

    als_config = {
        "factors": 64,
        "regularization": 0.05,
        "iterations": 10,
        "seed": 42,
        "kappa": 40.0,
        "train_time_sec": als_train_time,
        "avg_epoch_time_sec": als_train_time / 10.0,
        "eval_time_sec": als_eval_time,
        "peak_mem_mb": peak_mem / (1024 * 1024),
    }
    (als_dir / "config.json").write_text(json.dumps(als_config, indent=2), encoding="utf-8")
    (als_dir / "metrics.json").write_text(json.dumps(als_metrics, indent=2), encoding="utf-8")
    (als_dir / "per_user_metrics.json").write_text(json.dumps(als_user_metrics, indent=2), encoding="utf-8")

    print("\nImplicit MSVD Test Results:")
    for k in [5, 10, 20]:
        print(f"  Recall@{k:2d}: {als_metrics[f'Recall@{k}']:.4f} | NDCG@{k:2d}: {als_metrics[f'NDCG@{k}']:.4f} | Precision@{k:2d}: {als_metrics[f'Precision@{k}']:.4f} | MAP@{k:2d}: {als_metrics[f'MAP@{k}']:.4f}")

    runtime_summary = {
        "dataset": "1k",
        "events": cnt,
        "train_matrix_shape": list(P_train_test.shape),
        "popularity_fit_sec": pop_train_time,
        "popularity_eval_sec": pop_eval_time,
        "als_train_sec": als_train_time,
        "als_epoch_avg_sec": als_train_time / 10.0,
        "als_eval_sec": als_eval_time,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")

    print("\n=================================================================")
    print("PHASE 5 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
