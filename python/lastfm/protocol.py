"""Phase 4 Hyperparameter search and validation-to-test protocol runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np

from .als import ImplicitMSVD
from .baseline import PopularityRecommender
from .evaluate import evaluate_model
from .split import split_train_val_test


def run_phase4_protocol(
    root: Path,
    dataset: str = "1k",
    train_cutoff: str = "2009-04-01T00:00:00Z",
    val_cutoff: str = "2009-05-01T00:00:00Z",
    test_end: Optional[str] = "2009-07-01T00:00:00Z",
    factors_grid: Sequence[int] = (32, 64, 128),
    reg_grid: Sequence[float] = (0.01, 0.05, 0.1, 0.5),
    iterations_grid: Sequence[int] = (10, 15, 20),
    seed: int = 42,
    kappa: float = 40.0,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    results_dir = (out_dir or (root / "results/phase4")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    val_dir = results_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir = results_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Performing 3-Way Chronological Split ===")
    splits = split_train_val_test(
        root,
        dataset=dataset,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        test_end=test_end,
        kappa=kappa,
    )

    val_split = splits["validation_split"]
    test_split = splits["test_split"]

    split_summary = {
        "dataset": dataset,
        "train_period": f"[2002-01-01, {train_cutoff})",
        "val_period": f"[{train_cutoff}, {val_cutoff})",
        "test_period": f"[{val_cutoff}, {test_end})",
        "train_shape": [val_split["n_users"], val_split["n_items"]],
        "train_nnz": int(val_split["P_train"].nnz),
        "val_evaluable_users": val_split["evaluable_users"],
        "trainval_shape": [test_split["n_users"], test_split["n_items"]],
        "trainval_nnz": int(test_split["P_train"].nnz),
        "test_evaluable_users": test_split["evaluable_users"],
    }
    (results_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    print("\n=== Step 2: Running Hyperparameter Search on VALIDATION ONLY ===")
    P_train_val = val_split["P_train"]
    D_train_val = val_split["D_train"]
    val_positives = val_split["val_positives"]

    validation_results: List[Dict[str, Any]] = []
    best_config: Optional[Dict[str, Any]] = None
    best_ndcg10: float = -1.0

    total_grid_runs = len(factors_grid) * len(reg_grid) * len(iterations_grid)
    run_idx = 0

    for factors in factors_grid:
        for reg in reg_grid:
            for iters in iterations_grid:
                run_idx += 1
                t0 = time.perf_counter()
                model = ImplicitMSVD(factors=factors, regularization=reg, iterations=iters, seed=seed)
                model.fit(P_train_val, D_train_val, compute_loss_history=False, verbose=False)
                t_train = time.perf_counter() - t0

                t_eval_start = time.perf_counter()
                macro_m, _ = evaluate_model(model, P_train_val, val_positives, k_list=[5, 10, 20], exclude_seen=True)
                t_eval = time.perf_counter() - t_eval_start

                record = {
                    "factors": factors,
                    "regularization": reg,
                    "iterations": iters,
                    "seed": seed,
                    "train_time_sec": t_train,
                    "eval_time_sec": t_eval,
                    **macro_m,
                }
                validation_results.append(record)

                ndcg10 = macro_m["NDCG@10"]
                print(f"[{run_idx:2d}/{total_grid_runs:2d}] factors={factors:3d}, reg={reg:4.2f}, iters={iters:2d} -> Val NDCG@10: {ndcg10:.4f}, Recall@10: {macro_m['Recall@10']:.4f} ({t_train:.2f}s)")

                # Selection rule: Highest NDCG@10 on validation, tie-break on simpler (lower factors, lower iters)
                if ndcg10 > best_ndcg10:
                    best_ndcg10 = ndcg10
                    best_config = {
                        "factors": factors,
                        "regularization": reg,
                        "iterations": iters,
                        "seed": seed,
                        "val_ndcg10": ndcg10,
                        "selection_metric": "NDCG@10 on validation split",
                    }

    (val_dir / "hyperparameter_results.json").write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    (val_dir / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\nBest configuration selected from Validation: {best_config}")

    print("\n=== Step 3: Running Final Unbiased Evaluation on HELD-OUT TEST ===")
    P_train_test = test_split["P_train"]
    D_train_test = test_split["D_train"]
    Counts_train_test = test_split["Counts_train"]
    test_positives = test_split["test_positives"]

    # 1. Popularity Baseline on Test
    print("Evaluating Popularity Baseline on Test...")
    pop = PopularityRecommender().fit(Counts_train_test)
    pop_metrics, pop_user_metrics = evaluate_model(pop, P_train_test, test_positives, k_list=[5, 10, 20], exclude_seen=True)

    # 2. Selected ImplicitMSVD on Test
    assert best_config is not None
    print(f"Training Selected ImplicitMSVD (factors={best_config['factors']}, reg={best_config['regularization']}, iters={best_config['iterations']}) on Full Training Data...")
    t_als_start = time.perf_counter()
    selected_model = ImplicitMSVD(
        factors=best_config["factors"],
        regularization=best_config["regularization"],
        iterations=best_config["iterations"],
        seed=seed,
    )
    selected_model.fit(P_train_test, D_train_test, compute_loss_history=False, verbose=False)
    als_train_time = time.perf_counter() - t_als_start

    t_als_eval = time.perf_counter()
    als_metrics, als_user_metrics = evaluate_model(selected_model, P_train_test, test_positives, k_list=[5, 10, 20], exclude_seen=True)
    als_eval_time = time.perf_counter() - t_als_eval

    final_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selected_config": best_config,
        "test_evaluable_users": test_split["evaluable_users"],
        "popularity_metrics": pop_metrics,
        "als_metrics": als_metrics,
        "als_train_time_sec": als_train_time,
        "als_eval_time_sec": als_eval_time,
    }

    (test_dir / "final_results.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    (test_dir / "als_per_user_metrics.json").write_text(json.dumps(als_user_metrics, indent=2), encoding="utf-8")
    (test_dir / "pop_per_user_metrics.json").write_text(json.dumps(pop_user_metrics, indent=2), encoding="utf-8")

    return final_results
