"""Phase 8 Multi-Stream Arbiter and Adaptive Re-Ranking full-scale experiment runner."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.arbiter import MultiStreamArbiter, normalize_scores
from lastfm.evaluate import evaluate_model
from lastfm.ingest import connect
from lastfm.split import split_train_val_test
from lastfm.tag_msvd import TagFusedMSVD, build_tag_matrix
from lastfm.temporal import TemporalMSVD, compute_temporal_affinity_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8")


def main():
    root = Path.cwd()
    out_dir = root / "results/phase8"
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
    print("PHASE 8: FULL-SCALE MULTI-STREAM ARBITER EXECUTION")
    print("=================================================================")

    # 1. Dataset splits
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
    T_val, _ = build_tag_matrix(root, val_item_ids, meta_parquet, min_tag_freq=5)
    T_test, _ = build_tag_matrix(root, test_item_ids, meta_parquet, min_tag_freq=5)
    t_tag = time.perf_counter() - t0_tag

    # 3. Build Temporal Matrices (Stream B)
    print("\n--- 3. Constructing Sparse Temporal Matrices (gamma=0.1) ---")
    t0_temp = time.perf_counter()
    S_val = compute_temporal_affinity_matrix(
        root, user_ids=val_user_ids, item_ids=val_item_ids, cutoff_iso="2009-04-01T00:00:00Z", gamma=0.1, kappa=40.0
    )
    S_test = compute_temporal_affinity_matrix(
        root, user_ids=test_user_ids, item_ids=test_item_ids, cutoff_iso="2009-05-01T00:00:00Z", gamma=0.1, kappa=40.0
    )
    t_temp = time.perf_counter() - t0_temp

    # 4. Train base ALS models
    print("\n--- 4. Fitting Base MSVD Models ---")
    t0_val_als = time.perf_counter()
    val_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    val_msvd.fit(val_split["P_train"], val_split["D_train"], compute_loss_history=False, verbose=True)
    t_val_als = time.perf_counter() - t0_val_als

    # 5. Validation Grid Search over Normalization, Gating Strategy & Weights
    print("\n--- 5. Validation Search for Arbiter Configuration ---")
    val_results = []
    norm_methods = ["minmax", "zscore", "raw"]
    weight_candidates = [
        [0.6, 0.2, 0.2],
        [0.4, 0.3, 0.3],
        [0.3, 0.2, 0.5],
        [0.2, 0.3, 0.5],
        [0.5, 0.1, 0.4],
    ]
    gating_modes = ["global", "user_aware"]

    best_val_ndcg = -1.0
    best_config = {}

    for norm_m in norm_methods:
        for w in weight_candidates:
            for gm in gating_modes:
                arb = MultiStreamArbiter(
                    base_msvd=val_msvd,
                    tag_matrix=T_val,
                    temporal_matrix=S_val,
                    weights=w,
                    normalization=norm_m,
                    gating_mode=gm,
                    temperature=1.0,
                )
                arb.fit_user_profiles(val_split["P_train"], val_split["D_train"])
                m, _ = evaluate_model(arb, val_split["P_train"], val_split["val_positives"], k_list=[5, 10, 20], exclude_seen=True)
                
                rec = {
                    "normalization": norm_m,
                    "weights": w,
                    "gating_mode": gm,
                    "Recall@10": m["Recall@10"],
                    "Precision@10": m["Precision@10"],
                    "NDCG@10": m["NDCG@10"],
                    "MAP@10": m["MAP@10"],
                }
                val_results.append(rec)
                print(f"  norm={norm_m:6s}, gm={gm:10s}, w={w} -> Val NDCG@10: {m['NDCG@10']:.6f} | Recall@10: {m['Recall@10']:.6f}")

                if m["NDCG@10"] > best_val_ndcg:
                    best_val_ndcg = m["NDCG@10"]
                    best_config = rec

    (val_dir / "hyperparameter_results.json").write_text(json.dumps(val_results, indent=2), encoding="utf-8")
    (val_dir / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\nSelected Configuration on Validation: {best_config}")

    # 6. Fit Final Test Base Model on TRAIN+VAL
    print("\n--- 6. Fitting Final Full-Scale MSVD Model on TRAIN+VAL ---")
    t0_test_als = time.perf_counter()
    test_msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    test_msvd.fit(test_split["P_train"], test_split["D_train"], compute_loss_history=False, verbose=True)
    t_test_als = time.perf_counter() - t0_test_als

    # 7. Final Test Evaluation across 8 Comparative Baselines & Arbiters
    print("\n--- 7. Executing 8-Way Final Test Comparative Evaluation ---")
    # 1. Popularity
    from lastfm.baseline import PopularityRecommender
    pop = PopularityRecommender()
    pop.fit(test_split["Counts_train"])
    m_pop, _ = evaluate_model(pop, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 2. Base MSVD
    m_msvd, _ = evaluate_model(test_msvd, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 3. Tag-Fused MSVD (Stream A)
    tag_model = TagFusedMSVD(test_msvd, T_test, beta=1.0)
    tag_model.fit_user_profiles(test_split["P_train"], test_split["D_train"])
    m_tag, _ = evaluate_model(tag_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 4. Temporal MSVD (Stream B)
    temp_model = TemporalMSVD(test_msvd, S_test, alpha=1.0, tag_matrix=None, beta=0.0)
    m_temp, _ = evaluate_model(temp_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 5. Fixed Tag + Temporal MSVD
    fixed_model = TemporalMSVD(test_msvd, S_test, alpha=1.0, tag_matrix=T_test, beta=1.0)
    fixed_model.fit_user_tag_profiles(test_split["P_train"], test_split["D_train"])
    m_fixed, _ = evaluate_model(fixed_model, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 6. Global Normalized Fusion Arbiter
    global_arbiter = MultiStreamArbiter(
        base_msvd=test_msvd,
        tag_matrix=T_test,
        temporal_matrix=S_test,
        weights=best_config["weights"],
        normalization=best_config["normalization"],
        gating_mode="global",
    )
    global_arbiter.fit_user_profiles(test_split["P_train"], test_split["D_train"])
    m_global, _ = evaluate_model(global_arbiter, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)

    # 7. Adaptive User-Aware Arbiter
    user_arbiter = MultiStreamArbiter(
        base_msvd=test_msvd,
        tag_matrix=T_test,
        temporal_matrix=S_test,
        weights=best_config["weights"],
        normalization=best_config["normalization"],
        gating_mode="user_aware",
    )
    user_arbiter.fit_user_profiles(test_split["P_train"], test_split["D_train"])
    t0_eval = time.perf_counter()
    m_user_arb, per_user_records = evaluate_model(user_arbiter, test_split["P_train"], test_split["test_positives"], k_list=[5, 10, 20], exclude_seen=True)
    t_eval = time.perf_counter() - t0_eval

    final_metrics = {
        "Popularity": m_pop,
        "Implicit_MSVD": m_msvd,
        "Tag_Fused_MSVD": m_tag,
        "Temporal_MSVD": m_temp,
        "Fixed_Tag_Temporal_MSVD": m_fixed,
        "Global_Normalized_Arbiter": m_global,
        "Adaptive_User_Arbiter": m_user_arb,
    }

    (test_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (test_dir / "per_user_metrics.json").write_text(json.dumps(per_user_records, indent=2), encoding="utf-8")

    # 8. User Activity Tier Breakdown
    print("\n--- 8. User Activity Tier Breakdown ---")
    user_train_counts = np.asarray(test_split["Counts_train"].sum(axis=1)).ravel()
    eval_user_indices = [rec["user_index"] for rec in per_user_records]
    eval_counts = [user_train_counts[u] for u in eval_user_indices]

    q33, q66 = np.percentile(eval_counts, [33.3, 66.6])
    bucket_results = {}

    for b_name, (low_c, high_c) in [
        ("low_activity", (0, q33)),
        ("mid_activity", (q33, q66)),
        ("high_activity", (q66, float("inf"))),
    ]:
        u_in_bucket = [
            rec for rec in per_user_records
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

    # 9. Arbiter Weight Distribution Analysis
    weights_list = [user_arbiter.get_user_weights(u).tolist() for u in eval_user_indices]
    w_arr = np.array(weights_list)
    arbiter_weights_summary = {
        "mean_weights": {"w_msvd": float(np.mean(w_arr[:, 0])), "w_tag": float(np.mean(w_arr[:, 1])), "w_temporal": float(np.mean(w_arr[:, 2]))},
        "median_weights": {"w_msvd": float(np.median(w_arr[:, 0])), "w_tag": float(np.median(w_arr[:, 1])), "w_temporal": float(np.median(w_arr[:, 2]))},
        "std_weights": {"w_msvd": float(np.std(w_arr[:, 0])), "w_tag": float(np.std(w_arr[:, 1])), "w_temporal": float(np.std(w_arr[:, 2]))},
    }
    (test_dir / "arbiter_weights.json").write_text(json.dumps(arbiter_weights_summary, indent=2), encoding="utf-8")

    # 10. Diversity & Stream Overlap Analysis
    print("\n--- 10. Diversity & Stream Overlap Analysis ---")
    all_recs_msvd = []
    all_recs_tag = []
    all_recs_temp = []
    all_recs_arb = []

    for u in eval_user_indices:
        seen = set(test_split["P_train"].indices[test_split["P_train"].indptr[u]:test_split["P_train"].indptr[u+1]])
        r_m = [i for i, _ in test_msvd.recommend(u, k=20, exclude_seen=True, seen_items=seen)]
        r_t = [i for i, _ in tag_model.recommend(u, k=20, exclude_seen=True, seen_items=seen)]
        r_r = [i for i, _ in temp_model.recommend(u, k=20, exclude_seen=True, seen_items=seen)]
        r_a = [i for i, _ in user_arbiter.recommend(u, k=20, exclude_seen=True, seen_items=seen)]
        all_recs_msvd.append(set(r_m))
        all_recs_tag.append(set(r_t))
        all_recs_temp.append(set(r_r))
        all_recs_arb.append(set(r_a))

    def jaccard(l1, l2):
        return float(np.mean([len(s1 & s2) / max(1, len(s1 | s2)) for s1, s2 in zip(l1, l2)]))

    stream_overlap = {
        "Jaccard_Arbiter_MSVD": jaccard(all_recs_arb, all_recs_msvd),
        "Jaccard_Arbiter_Tag": jaccard(all_recs_arb, all_recs_tag),
        "Jaccard_Arbiter_Temporal": jaccard(all_recs_arb, all_recs_temp),
        "Jaccard_MSVD_Tag": jaccard(all_recs_msvd, all_recs_tag),
        "Jaccard_MSVD_Temporal": jaccard(all_recs_msvd, all_recs_temp),
        "Jaccard_Tag_Temporal": jaccard(all_recs_tag, all_recs_temp),
    }
    (analysis_dir / "stream_overlap.json").write_text(json.dumps(stream_overlap, indent=2), encoding="utf-8")

    # Catalog coverage
    unique_rec_tracks = len(set.union(*all_recs_arb))
    diversity_metrics = {
        "total_recommendations": 20 * len(eval_user_indices),
        "unique_tracks_recommended_top20": unique_rec_tracks,
        "catalog_coverage_pct": (unique_rec_tracks / float(len(test_item_ids))) * 100.0,
    }
    (test_dir / "diversity_metrics.json").write_text(json.dumps(diversity_metrics, indent=2), encoding="utf-8")

    # Cold start summary
    has_tag = np.diff(T_test.indptr) > 0
    cold_start_metrics = {
        "total_catalog_items": len(test_item_ids),
        "tag_covered_items": int(np.sum(has_tag)),
        "evaluable_test_users": len(eval_user_indices),
    }
    (test_dir / "cold_start_metrics.json").write_text(json.dumps(cold_start_metrics, indent=2), encoding="utf-8")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_total = time.perf_counter() - t_start_total

    runtime_data = {
        "tag_build_time_sec": t_tag,
        "temp_build_time_sec": t_temp,
        "val_als_time_sec": t_val_als,
        "test_als_time_sec": t_test_als,
        "evaluation_time_sec": t_eval,
        "total_runtime_sec": t_total,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime_data, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print("\n=================================================================")
    print("PHASE 8 MULTI-STREAM ARBITER TEST ABLATION RESULTS (518 Evaluated Users):")
    print("=================================================================")
    header = f"{'Model':<28} | {'Recall@10':<10} | {'Prec@10':<10} | {'NDCG@10':<10} | {'MAP@10':<10} | {'NDCG@20':<10}"
    print(header)
    print("-" * len(header))
    for m_label, m_dict in final_metrics.items():
        print(f"{m_label:<28} | {m_dict['Recall@10']:<10.6f} | {m_dict['Precision@10']:<10.6f} | {m_dict['NDCG@10']:<10.6f} | {m_dict['MAP@10']:<10.6f} | {m_dict['NDCG@20']:<10.6f}")

    print(f"\nTotal Runtime: {t_total:.1f}s | Peak Memory: {peak_mem / (1024 * 1024):.1f} MB")
    print("=================================================================")
    print("PHASE 8 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
