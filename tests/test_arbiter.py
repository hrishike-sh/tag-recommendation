"""Unit tests for Phase 8 Multi-Stream Arbiter."""

import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.arbiter import MultiStreamArbiter, normalize_scores


def test_normalize_scores_minmax():
    """Verify min-max score normalization bounds to [0, 1]."""
    scores = np.array([-2.0, 0.0, 2.0, -np.inf], dtype=np.float32)
    norm = normalize_scores(scores, method="minmax")
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0
    assert np.isneginf(norm[3])


def test_normalize_scores_constant():
    """Verify constant scores normalize safely without division by zero."""
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    norm = normalize_scores(scores, method="minmax")
    assert np.all(norm == 0.0)


def test_arbiter_weights_sum_to_one():
    """Verify arbiter weights are non-negative and sum to 1."""
    base = ImplicitMSVD(factors=4, regularization=0.01, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    conf = sp.csr_matrix([[1.0, 0], [0, 1.0]], dtype=np.float32)
    base.fit(pref, conf, compute_loss_history=False, verbose=False)

    arbiter = MultiStreamArbiter(base, weights=[0.6, 0.2, 0.2], gating_mode="user_aware")
    arbiter.fit_user_profiles(pref, conf)

    for u in range(2):
        w = arbiter.get_user_weights(u)
        assert np.all(w >= 0.0)
        assert pytest.approx(float(np.sum(w)), rel=1e-5) == 1.0


def test_arbiter_recommendation_determinism():
    """Verify deterministic ranking and seen-item exclusion."""
    base = ImplicitMSVD(factors=4, regularization=0.01, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 0, 0]], dtype=np.float32)
    conf = sp.csr_matrix([[1.0, 0, 0]], dtype=np.float32)
    base.fit(pref, conf, compute_loss_history=False, verbose=False)

    tag_mat = sp.csr_matrix([[1.0], [0.5], [0.0]], dtype=np.float32)
    temp_mat = sp.csr_matrix([[0.0, 1.0, 0.0]], dtype=np.float32)

    arbiter = MultiStreamArbiter(base, tag_matrix=tag_mat, temporal_matrix=temp_mat, weights=[0.33, 0.33, 0.34])
    arbiter.fit_user_profiles(pref, conf)

    recs1 = arbiter.recommend(0, k=2, exclude_seen=True, seen_items={0})
    recs2 = arbiter.recommend(0, k=2, exclude_seen=True, seen_items={0})

    assert recs1 == recs2
    assert 0 not in [item for item, _ in recs1]
