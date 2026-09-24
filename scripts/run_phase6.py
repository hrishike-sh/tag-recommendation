"""Run Phase 6 validation search for beta and execute full-scale Tag-Fused MSVD evaluation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc
import duckdb
import numpy as np
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.evaluate import evaluate_model
from lastfm.ingest import connect
from lastfm.split import split_train_val_test
from lastfm.tag_msvd import TagFusedMSVD, build_tag_matrix


def main():
    root = Path.cwd()
    out_dir = root / "results/phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t_start_total = time.perf_counter()

    print("=================================================================")
    print("PHASE 6: FULL-SCALE TAG-FUSED MSVD (STREAM A) EXECUTION")
    print("=================================================================")

    # 1. Split datasets
    print("\n--- 1. Loading Chronological Split Data ---")
    splits = split_train_val_test(
        root,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        test_end="2009-07-01T00:00:00Z",
        kappa=40.0,
    )
    val_split = splits["validation_split"]
    test_split = splits["test_split"]

    meta_parquet = root / "data/lake/1k/529b8f83aada-3f0122fb/track_metadata.parquet"

    val_item_ids = val_split["item_ids"]
    test_item_ids = test_split["item_ids"]

    print("\n--- 2. Constructing Sparse TF-IDF Tag Matrices ---")
    t0_tag = time.perf_counter()
    T_val, tag_vocab_val = build_tag_matrix(root, val_item_ids, meta_parquet, min_tag_freq=5)
    T_test, tag_vocab_test = build_tag_matrix(root, test_item_ids, meta_parquet, min_tag_freq=5)
    t_tag = time.perf_counter() - t0_tag
    print(f"Tag matrices built in {t_tag:.2f}s.")
    print(f"Validation Tag Matrix: {T_val.shape[0]} items x {T_val.shape[1]} tags, NNZ: {T_val.nnz:,}")
    print(f"Test Tag Matrix: {T_test.shape[0]} items x {T_test.shape[1]} tags, NNZ: {T_test.nnz:,}")

    tag_summary = {
        "tag_vocab_size": T_test.shape[1],
        "test_tag_matrix_nnz": int(T_test.nnz),
        "tag_build_time_sec": t_tag,
    }
    (out_dir / "tag_summary.json").write_text(json.dumps(tag_summary, indent=2), encoding="utf-8")

    # 3. Fit base ALS models on Validation split (3 iterations for fast validation sweep, seed=42)
    print("\n--- 3. Fitting Validation Base MSVD ---")
    t0_val_als = time.perf_counter()
    val_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    val_msvd.fit(val_split["P_train"], val_split["D_train"], compute_loss_history=False, verbose=True)
    t_val_als = time.perf_counter() - t0_val_als

    print("\n--- 4. Hyperparameter Search for Beta on VALIDATION PERIOD ---")
    beta_grid = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    val_results = []
    best_beta = 0.0
    best_ndcg = -1.0

    val_fused = TagFusedMSVD(val_msvd, T_val, beta=0.0)
    val_fused.fit_user_profiles(val_split["P_train"], val_split["D_train"])

    for b in beta_grid:
        val_fused.beta = b
        m, _ = evaluate_model(val_fused, val_split["P_train"], val_split["val_positives"], k_list=[5, 10, 20], exclude_seen=True)
        rec = {"beta": b, **m}
        val_results.append(rec)
        print(f"  beta = {b:4.2f} -> Val NDCG@10: {m['NDCG@10']:.6f} | Recall@10: {m['Recall@10']:.6f} | Precision@10: {m['Precision@10']:.6f}")
        if m["NDCG@10"] > best_ndcg:
            best_ndcg = m["NDCG@10"]
            best_beta = b

    best_config = {"best_beta": best_beta, "val_ndcg10": best_ndcg}
    (val_dir / "hyperparameter_results.json").write_text(json.dumps(val_results, indent=2), encoding="utf-8")
    (val_dir / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\nSelected Beta on Validation: {best_beta} (NDCG@10: {best_ndcg:.6f})")

    # 4. Final Full-Scale Test Evaluation
    print(f"\n--- 5. Evaluating Tag-Fused MSVD (beta={best_beta}) on Held-Out Test ---")
    t0_test_als = time.perf_counter()
    test_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    test_msvd.fit(test_split["P_train"], test_split["D_train"], compute_loss_history=False, verbose=True)
    t_test_als = time.perf_counter() - t0_test_als

    test_fused = TagFusedMSVD(test_msvd, T_test, beta=best_beta)
    test_fused.fit_user_profiles(test_split["P_train"], test_split["D_train"])

    t0_eval = time.perf_counter()
    test_metrics, test_user_metrics = evaluate_model(
        test_fused, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True
    )
    t_eval = time.perf_counter() - t0_eval

    (test_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (test_dir / "per_user_metrics.json").write_text(json.dumps(test_user_metrics, indent=2), encoding="utf-8")

    # Cold start analysis
    # Item tag coverage among test candidates
    has_tag = np.diff(T_test.indptr) > 0
    num_tag_covered_items = int(np.sum(has_tag))
    num_total_test_items = T_test.shape[0]

    cold_start_metrics = {
        "total_catalog_items": num_total_test_items,
        "tag_covered_items": num_tag_covered_items,
        "tag_coverage_pct": num_tag_covered_items / max(1, num_total_test_items) * 100.0,
        "evaluable_test_users": len(test_split["test_positives"]),
    }
    (test_dir / "cold_start_metrics.json").write_text(json.dumps(cold_start_metrics, indent=2), encoding="utf-8")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_total = time.perf_counter() - t_start_total

    runtime_data = {
        "tag_build_time_sec": t_tag,
        "val_als_time_sec": t_val_als,
        "test_als_time_sec": t_test_als,
        "eval_time_sec": t_eval,
        "total_time_sec": t_total,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime_data, indent=2), encoding="utf-8")

    config_data = {
        "factors": 64,
        "regularization": 0.05,
        "iterations": 10,
        "kappa": 40.0,
        "best_beta": best_beta,
        "tag_vocab_size": T_test.shape[1],
        "similarity_function": "cosine (L2 normalized TF-IDF dot product)",
    }
    (out_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    print("\nTag-Fused MSVD Test Results:")
    for k in [5, 10, 20]:
        print(f"  Recall@{k:2d}: {test_metrics[f'Recall@{k}']:.6f} | NDCG@{k:2d}: {test_metrics[f'NDCG@{k}']:.6f} | Precision@{k:2d}: {test_metrics[f'Precision@{k}']:.6f} | MAP@{k:2d}: {test_metrics[f'MAP@{k}']:.6f}")

    print(f"\nTotal Runtime: {t_total:.1f}s | Peak Memory: {peak_mem / (1024 * 1024):.1f} MB")
    print("\n=================================================================")
    print("PHASE 6 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()

