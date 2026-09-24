import json
import math
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD


def create_synthetic_data(kappa: float = 40.0):
    """Create a small synthetic dataset: 3 users, 4 items.

    Interactions:
      u0: i0 (count=5), i1 (count=2)
      u1: i1 (count=10), i2 (count=1)
      u2: i2 (count=3), i3 (count=8)
    """
    n_users = 3
    n_items = 4

    # (user, item, play_count)
    interactions = [
        (0, 0, 5),
        (0, 1, 2),
        (1, 1, 10),
        (1, 2, 1),
        (2, 2, 3),
        (2, 3, 8),
    ]

    rows = [u for u, i, c in interactions]
    cols = [i for u, i, c in interactions]
    counts = np.array([c for u, i, c in interactions], dtype=np.float32)

    # Preferences: 1 for observed
    prefs = np.ones(len(interactions), dtype=np.float32)
    # Confidence delta: kappa * ln(1 + count)
    conf_deltas = kappa * np.log1p(counts)

    P = sp.csr_matrix((prefs, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    D = sp.csr_matrix((conf_deltas, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    return P, D, interactions


def test_als_synthetic_training_and_dimensions():
    P, D, _ = create_synthetic_data(kappa=40.0)
    factors = 8
    regularization = 0.05
    iterations = 10

    model = ImplicitMSVD(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        seed=42,
    )
    model.fit(P, D, compute_loss_history=True, verbose=False)

    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.user_factors.shape == (3, factors)
    assert model.item_factors.shape == (4, factors)
    assert not np.isnan(model.user_factors).any()
    assert not np.isinf(model.user_factors).any()
    assert not np.isnan(model.item_factors).any()
    assert not np.isinf(model.item_factors).any()

    # Verify loss decreases
    assert len(model.history) == iterations
    losses = [h["loss"] for h in model.history]
    assert losses[-1] < losses[0]


def test_als_recommendations_and_seen_exclusion():
    P, D, _ = create_synthetic_data(kappa=40.0)
    model = ImplicitMSVD(factors=4, regularization=0.01, iterations=15, seed=42)
    model.fit(P, D, verbose=False)

    # For user 0, seen items are {0, 1}
    seen_u0 = {0, 1}
    recs_exclude = model.recommend(0, k=2, exclude_seen=True, seen_items=seen_u0)
    assert len(recs_exclude) <= 2
    for item_idx, score in recs_exclude:
        assert item_idx not in seen_u0
        assert item_idx in {2, 3}
        assert math.isfinite(score)

    # When not excluding seen
    recs_all = model.recommend(0, k=4, exclude_seen=False)
    assert len(recs_all) == 4
    all_indices = [idx for idx, _ in recs_all]
    assert set(all_indices) == {0, 1, 2, 3}


def test_als_save_and_load(tmp_path):
    P, D, _ = create_synthetic_data(kappa=40.0)
    model = ImplicitMSVD(factors=6, regularization=0.02, iterations=5, seed=123)
    model.fit(P, D, verbose=False)

    save_dir = tmp_path / "model_export"
    model.save(save_dir)

    assert (save_dir / "user_factors.npy").exists()
    assert (save_dir / "item_factors.npy").exists()
    assert (save_dir / "model_config.json").exists()

    loaded_model = ImplicitMSVD.load(save_dir)
    assert loaded_model.factors == model.factors
    assert loaded_model.regularization == model.regularization
    assert loaded_model.iterations == model.iterations
    assert loaded_model.seed == model.seed

    np.testing.assert_allclose(loaded_model.user_factors, model.user_factors, rtol=1e-6)
    np.testing.assert_allclose(loaded_model.item_factors, model.item_factors, rtol=1e-6)

    recs_orig = model.recommend(1, k=3, exclude_seen=False)
    recs_loaded = loaded_model.recommend(1, k=3, exclude_seen=False)
    assert recs_orig == recs_loaded


def test_als_deterministic_seed_reproducibility():
    P, D, _ = create_synthetic_data()

    m1 = ImplicitMSVD(factors=4, regularization=0.1, iterations=5, seed=777)
    m1.fit(P, D, verbose=False)

    m2 = ImplicitMSVD(factors=4, regularization=0.1, iterations=5, seed=777)
    m2.fit(P, D, verbose=False)

    np.testing.assert_array_equal(m1.user_factors, m2.user_factors)
    np.testing.assert_array_equal(m1.item_factors, m2.item_factors)

    # Different seed gives different results
    m3 = ImplicitMSVD(factors=4, regularization=0.1, iterations=5, seed=888)
    m3.fit(P, D, verbose=False)
    assert not np.allclose(m1.user_factors, m3.user_factors)


def test_als_zero_confidence_delta():
    """When confidence_delta is all zeros, confidence is 1 everywhere."""
    n_users, n_items = 3, 3
    P = sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(n_users, n_items), dtype=np.float32)
    D = sp.csr_matrix((n_users, n_items), dtype=np.float32)  # all zeros

    model = ImplicitMSVD(factors=4, regularization=0.1, iterations=5, seed=42)
    model.fit(P, D, verbose=False)

    assert not np.isnan(model.user_factors).any()
    assert not np.isnan(model.item_factors).any()


def test_als_zero_interaction_user_and_item():
    """Ensure zero-interaction users and items get zero vectors and don't fail."""
    n_users, n_items = 4, 4
    # User 3 and Item 3 have no interactions
    P = sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(n_users, n_items), dtype=np.float32)
    D = sp.csr_matrix(([10.0, 10.0], ([0, 1], [0, 1])), shape=(n_users, n_items), dtype=np.float32)

    model = ImplicitMSVD(factors=4, regularization=0.05, iterations=5, seed=42)
    model.fit(P, D, verbose=False)

    # User 3 (no rows) should have factor vector = 0
    np.testing.assert_array_equal(model.user_factors[3], np.zeros(4))
    # Item 3 (no cols) should have factor vector = 0
    np.testing.assert_array_equal(model.item_factors[3], np.zeros(4))
