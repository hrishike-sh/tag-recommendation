"""Phase 7 full-scale temporal and nostalgia extension experiment runner."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.evaluate import evaluate_model
from lastfm.ingest import connect
from lastfm.split import split_train_val_test
from lastfm.tag_msvd import TagFusedMSVD, build_tag_matrix
from lastfm.temporal import TemporalMSVD, compute_temporal_affinity_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7")


def main():
    root = Path.cwd()
    out_dir = root / "results/phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t_start_total = time.perf_counter()

    print("=================================================================")
    print("PHASE 7: FULL-SCALE TEMPORAL & NOSTALGIA MSVD (STREAM B) RUNNER")
    print("=================================================================")

    # 1. Load chronological dataset splits
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

    val_user_ids = val_split["user_ids"]
    val_item_ids = val_split["item_ids"]
    test_user_ids = test_split["user_ids"]
    test_item_ids = test_split["item_ids"]

    # Dataset summary artifact
    dataset_summary = {
        "dataset": "Last.fm 1K",
        "clean_events": 19150865,
        "total_users": 992,
        "unique_tracks": 1505185,
        "train_catalog_items": len(test_item_ids),
        "train_cutoff": splits["train_cutoff"],
        "val_cutoff": splits["val_cutoff"],
        "test_end": splits["test_end"],
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    # 2. Build Tag Matrices (Stream A)
    print("\n--- 2. Constructing Sparse TF-IDF Tag Matrices ---")
    t0_tag = time.perf_counter()
    T_val, tag_vocab_val = build_tag_matrix(root, val_item_ids, meta_parquet, min_tag_freq=5)
    T_test, tag_vocab_test = build_tag_matrix(root, test_item_ids, meta_parquet, min_tag_freq=5)
    t_tag = time.perf_counter() - t0_tag
    print(f"Tag matrices constructed in {t_tag:.2f}s.")

    # 3. Train base ALS models
    print("\n--- 3. Fitting Validation Base MSVD Model ---")
    t0_val_als = time.perf_counter()
    val_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    val_msvd.fit(val_split["P_train"], val_split["D_train"], compute_loss_history=False, verbose=True)
    t_val_als = time.perf_counter() - t0_val_als

    # 4. Hyperparameter Search for (gamma, alpha) on VALIDATION PERIOD
    print("\n--- 4. Grid Search for Gamma and Alpha on VALIDATION PERIOD ---")
    gamma_grid = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    alpha_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    val_results = []
    best_gamma = 0.0
    best_alpha = 0.0
    best_val_ndcg = -1.0
    best_val_recall = -1.0

    t0_val_grid = time.perf_counter()
    for g in gamma_grid:
        # Build validation temporal matrix strictly before 2009-04-01
        S_val = compute_temporal_affinity_matrix(
            root,
            user_ids=val_user_ids,
            item_ids=val_item_ids,
            cutoff_iso="2009-04-01T00:00:00Z",
            gamma=g,
            mode="recency",
            track_meta_parquet=meta_parquet,
            kappa=40.0,
        )

        val_temporal_model = TemporalMSVD(
            base_msvd=val_msvd,
            temporal_matrix=S_val,
            alpha=0.0,
            tag_matrix=T_val,
            beta=1.0,
        )
        val_temporal_model.fit_user_tag_profiles(val_split["P_train"], val_split["D_train"])

        for a in alpha_grid:
            val_temporal_model.alpha = a
            metrics, _ = evaluate_model(
                val_temporal_model,
                val_split["P_train"],
                val_split["val_positives"],
                k_list=[5, 10, 20],
                exclude_seen=True,
            )
            rec = {
                "gamma": g,
                "alpha": a,
                "Recall@10": metrics["Recall@10"],
                "NDCG@10": metrics["NDCG@10"],
                "Precision@10": metrics["Precision@10"],
                "MAP@10": metrics["MAP@10"],
            }
            val_results.append(rec)
            print(f"  gamma={g:6.4f}, alpha={a:4.2f} -> Val NDCG@10: {metrics['NDCG@10']:.6f} | Recall@10: {metrics['Recall@10']:.6f}")

            if metrics["NDCG@10"] > best_val_ndcg:
                best_val_ndcg = metrics["NDCG@10"]
                best_val_recall = metrics["Recall@10"]
                best_gamma = g
                best_alpha = a

    t_val_grid = time.perf_counter() - t0_val_grid

    best_config = {
        "best_gamma": best_gamma,
        "best_alpha": best_alpha,
        "val_ndcg10": best_val_ndcg,
        "val_recall10": best_val_recall,
        "beta": 1.0,
        "factors": 64,
        "regularization": 0.05,
        "kappa": 40.0,
        "iterations": 10,
    }
    (val_dir / "hyperparameter_results.json").write_text(json.dumps(val_results, indent=2), encoding="utf-8")
    (val_dir / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\nSelected Configuration on Validation: gamma={best_gamma}, alpha={best_alpha} (NDCG@10: {best_val_ndcg:.6f})")

    # 5. Build Final Temporal Matrix on TRAIN+VAL (strictly before 2009-05-01)
    print("\n--- 5. Constructing Test Temporal Affinity Matrix (TRAIN+VAL) ---")
    t0_test_temp = time.perf_counter()
    S_test = compute_temporal_affinity_matrix(
        root,
        user_ids=test_user_ids,
        item_ids=test_item_ids,
        cutoff_iso="2009-05-01T00:00:00Z",
        gamma=best_gamma,
        mode="recency",
        track_meta_parquet=meta_parquet,
        kappa=40.0,
    )
    t_test_temp = time.perf_counter() - t0_test_temp

    # 6. Fit Final Base MSVD Model on TRAIN+VAL
    print("\n--- 6. Fitting Final Full-Scale MSVD Model on TRAIN+VAL ---")
    t0_test_als = time.perf_counter()
    test_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    test_msvd.fit(test_split["P_train"], test_split["D_train"], compute_loss_history=False, verbose=True)
    t_test_als = time.perf_counter() - t0_test_als

    # 7. Execute 5-Way Final Test Ablations on Held-Out Test
    print("\n--- 7. Executing 5-Way Final Test Ablations ---")
    # A. Popularity Baseline
    from lastfm.baseline import PopularityRecommender
    pop_model = PopularityRecommender()
    pop_model.fit(test_split["Counts_train"])
    pop_metrics, _ = evaluate_model(pop_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # B. Pure Implicit MSVD
    msvd_metrics, _ = evaluate_model(test_msvd, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # C. Tag-Fused MSVD (Stream A)
    tag_model = TagFusedMSVD(test_msvd, T_test, beta=1.0)
    tag_model.fit_user_profiles(test_split["P_train"], test_split["D_train"])
    tag_metrics, _ = evaluate_model(tag_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # D. Temporal MSVD (Stream B only)
    temp_only_model = TemporalMSVD(test_msvd, S_test, alpha=best_alpha, tag_matrix=None, beta=0.0)
    temp_only_metrics, _ = evaluate_model(temp_only_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # E. Full Tag + Temporal MSVD (Stream A + Stream B)
    full_model = TemporalMSVD(test_msvd, S_test, alpha=best_alpha, tag_matrix=T_test, beta=1.0)
    full_model.fit_user_tag_profiles(test_split["P_train"], test_split["D_train"])
    t0_eval = time.perf_counter()
    full_metrics, per_user_metrics = evaluate_model(full_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)
    t_eval = time.perf_counter() - t0_eval

    ablation_metrics = {
        "Popularity": pop_metrics,
        "Implicit_MSVD": msvd_metrics,
        "Tag_Fused_MSVD": tag_metrics,
        "Temporal_MSVD": temp_only_metrics,
        "Tag_Temporal_MSVD": full_metrics,
    }

    (test_dir / "metrics.json").write_text(json.dumps(ablation_metrics, indent=2), encoding="utf-8")
    (test_dir / "per_user_metrics.json").write_text(json.dumps(per_user_metrics, indent=2), encoding="utf-8")

    # 8. Temporal User Bucket Analysis
    print("\n--- 8. Computing User Activity & History Bucket Metrics ---")
    user_train_counts = np.asarray(test_split["Counts_train"].sum(axis=1)).ravel()
    eval_user_indices = [rec["user_index"] for rec in per_user_metrics]
    eval_counts = [user_train_counts[u] for u in eval_user_indices]

    q33, q66 = np.percentile(eval_counts, [33.3, 66.6])
    bucket_results = {"low_activity": {}, "mid_activity": {}, "high_activity": {}}

    for b_name, (low_c, high_c) in [
        ("low_activity", (0, q33)),
        ("mid_activity", (q33, q66)),
        ("high_activity", (q66, float("inf"))),
    ]:
        u_in_bucket = [
            rec for rec in per_user_metrics
            if low_c <= user_train_counts[rec["user_index"]] < high_c
        ]
        n_b = len(u_in_bucket)
        if n_b > 0:
            b_metrics = {}
            for k in [5, 10, 20]:
                for m_name in ["Recall", "Precision", "NDCG", "MAP"]:
                    b_metrics[f"{m_name}@{k}"] = sum(r[f"{m_name}@{k}"] for r in u_in_bucket) / float(n_b)
            b_metrics["user_count"] = n_b
            bucket_results[b_name] = b_metrics

    (test_dir / "bucket_metrics.json").write_text(json.dumps(bucket_results, indent=2), encoding="utf-8")

    # 9. Recommendation Temporal Composition & Diversity Analysis
    print("\n--- 9. Computing Recommendation Composition & Diversity ---")
    # Inspect top-20 recommendations for recent vs historical artist affinity
    all_rec_items: List[int] = []
    top_k_artists: Set[str] = set()

    with connect(root) as db:
        item_to_artist_map = dict(
            db.execute(f"SELECT item_id, artist_id FROM read_parquet('{meta_parquet.as_posix()}')").fetchall()
        )

    for u in eval_user_indices:
        recs = full_model.recommend(u, k=20, exclude_seen=True, seen_items=set(test_split["P_train"].indices[test_split["P_train"].indptr[u]:test_split["P_train"].indptr[u+1]]))
        for i_idx, _ in recs:
            all_rec_items.append(i_idx)
            iid = test_item_ids[i_idx]
            if iid in item_to_artist_map:
                top_k_artists.add(item_to_artist_map[iid])

    unique_recommended_items = len(set(all_rec_items))
    catalog_coverage_pct = (unique_recommended_items / float(len(test_item_ids))) * 100.0

    composition_data = {
        "total_recommendations_generated": len(all_rec_items),
        "unique_items_recommended": unique_recommended_items,
        "unique_artists_recommended": len(top_k_artists),
        "catalog_coverage_percentage": catalog_coverage_pct,
    }
    (analysis_dir / "recommendation_temporal_composition.json").write_text(json.dumps(composition_data, indent=2), encoding="utf-8")

    # Cold start accounting
    has_tag = np.diff(T_test.indptr) > 0
    cold_start_metrics = {
        "catalog_size": len(test_item_ids),
        "tag_covered_items": int(np.sum(has_tag)),
        "evaluable_users": len(test_split["test_positives"]),
        "unique_items_recommended_top20": unique_recommended_items,
    }
    (test_dir / "cold_start_metrics.json").write_text(json.dumps(cold_start_metrics, indent=2), encoding="utf-8")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_total = time.perf_counter() - t_start_total

    runtime_data = {
        "tag_build_time_sec": t_tag,
        "val_als_time_sec": t_val_als,
        "val_grid_search_time_sec": t_val_grid,
        "test_temp_matrix_time_sec": t_test_temp,
        "test_als_time_sec": t_test_als,
        "evaluation_time_sec": t_eval,
        "total_runtime_sec": t_total,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime_data, indent=2), encoding="utf-8")

    (out_dir / "config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print("\n=================================================================")
    print("PHASE 7 TEST ABLATION RESULTS (Macro-Averaged over 518 Test Users):")
    print("=================================================================")
    header = f"{'Model':<22} | {'Recall@10':<10} | {'Prec@10':<10} | {'NDCG@10':<10} | {'MAP@10':<10} | {'NDCG@20':<10}"
    print(header)
    print("-" * len(header))
    for m_label, m_dict in ablation_metrics.items():
        print(f"{m_label:<22} | {m_dict['Recall@10']:<10.6f} | {m_dict['Precision@10']:<10.6f} | {m_dict['NDCG@10']:<10.6f} | {m_dict['MAP@10']:<10.6f} | {m_dict['NDCG@20']:<10.6f}")

    print(f"\nTotal Runtime: {t_total:.1f}s | Peak Memory: {peak_mem / (1024 * 1024):.1f} MB")
    print("=================================================================")
    print("PHASE 7 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
