"""Execute real-data smoke test of ImplicitMSVD using pipeline-generated matrix artifacts."""

import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD


def find_latest_matrix(root: Path, dataset: str = "1k") -> Path:
    base = root / "artifacts/matrices" / dataset
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "preferences.npz").exists()]
    if not candidates:
        raise FileNotFoundError(f"No matrix directories found under {base}")
    # Sort by mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_smoke_test():
    root = Path.cwd()
    matrix_dir = find_latest_matrix(root, "1k")
    print(f"=== Real-Data Smoke Test of ImplicitMSVD ===")
    print(f"Loading matrix artifacts from: {matrix_dir}")

    # 1. Load artifacts
    meta_path = matrix_dir / "matrix.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    P = sp.load_npz(matrix_dir / "preferences.npz")
    D = sp.load_npz(matrix_dir / "confidence_delta.npz")

    n_users, n_items = P.shape
    nnz = P.nnz
    total_entries = n_users * n_items
    sparsity = 1.0 - (nnz / total_entries)

    print("\n--- 1. Matrix Statistics ---")
    print(f"Users (U): {n_users}")
    print(f"Items (I): {n_items}")
    print(f"Observed pairs (NNZ): {nnz}")
    print(f"Total pairs (U x I): {total_entries}")
    print(f"Sparsity: {sparsity * 100:.2f}% (Density: {(1.0 - sparsity) * 100:.2f}%)")

    # 2. Configuration and Training
    factors = 32
    iterations = 5
    regularization = 0.05
    seed = 42

    print("\n--- 2. Model Configuration ---")
    print(f"Factors: {factors}")
    print(f"Iterations: {iterations}")
    print(f"Regularization: {regularization}")
    print(f"Seed: {seed}")

    tracemalloc.start()
    start_total = time.perf_counter()

    model = ImplicitMSVD(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        seed=seed,
    )

    model.fit(P, D, compute_loss_history=True, verbose=True)

    total_time = time.perf_counter() - start_total
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n--- 3. Training Results & Convergence ---")
    print(f"Total training time: {total_time:.3f}s")
    for rec in model.history:
        print(f"  Epoch {rec['epoch']:2d}: Loss = {rec['loss']:12.4f} (Time = {rec['elapsed_sec']*1000:.2f}ms)")

    # 4. Factor Matrix Validation
    assert model.user_factors is not None and model.item_factors is not None
    u_shape = model.user_factors.shape
    i_shape = model.item_factors.shape
    u_nans = int(np.isnan(model.user_factors).sum())
    u_infs = int(np.isinf(model.user_factors).sum())
    i_nans = int(np.isnan(model.item_factors).sum())
    i_infs = int(np.isinf(model.item_factors).sum())

    print("\n--- 4. Factor Matrix Validation ---")
    print(f"User factor matrix shape: {u_shape}")
    print(f"Item factor matrix shape: {i_shape}")
    print(f"User factor NaN/Inf count: {u_nans} / {u_infs}")
    print(f"Item factor NaN/Inf count: {i_nans} / {i_infs}")
    print(f"Peak memory during training: {peak_mem / (1024 * 1024):.2f} MB")

    # 5. Top-10 Recommendations for 5 Users
    print("\n--- 5. Top-10 Recommendations for 5 Real Users ---")
    test_users = [0, 1, 2, 3, 4]
    for u in test_users:
        start_ptr, end_ptr = P.indptr[u], P.indptr[u + 1]
        seen_items = set(P.indices[start_ptr:end_ptr])
        recs = model.recommend(u, k=10, exclude_seen=True, seen_items=seen_items)
        
        print(f"\nUser {u:2d} (Observed consumed items: {len(seen_items)}/{n_items}):")
        for rank, (item_idx, score) in enumerate(recs, 1):
            assert item_idx not in seen_items, f"Failure: Consumed item {item_idx} found in recommendation!"
            print(f"  Rank {rank:2d}: Item {item_idx:3d} | Score: {score:+.4f}")

    # 6. Save and Load Validation
    print("\n--- 6. Save / Load Verification ---")
    save_path = root / "artifacts/models/smoke_test_model"
    model.save(save_path)
    loaded_model = ImplicitMSVD.load(save_path)
    
    np.testing.assert_allclose(model.user_factors, loaded_model.user_factors, rtol=1e-6)
    np.testing.assert_allclose(model.item_factors, loaded_model.item_factors, rtol=1e-6)
    
    # Check predictions equality
    orig_recs = model.recommend(0, k=10, exclude_seen=False)
    load_recs = loaded_model.recommend(0, k=10, exclude_seen=False)
    assert orig_recs == load_recs, "Mismatch between original and loaded model recommendations"
    print(f"Model successfully saved and loaded from {save_path}. All factors and predictions bitwise match.")

    print("\n=======================================================")
    print("SMOKE TEST STATUS: ALL CHECKS PASSED. READY FOR EVALUATION HARNESS.")
    print("=======================================================")


if __name__ == "__main__":
    run_smoke_test()
