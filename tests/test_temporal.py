"""Comprehensive unit and leakage tests for Phase 7 Temporal / Nostalgia extension."""

from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.temporal import TemporalMSVD, compute_temporal_affinity_matrix


def test_temporal_msvd_initialization():
    """Verify TemporalMSVD properly initializes and maintains shape integrity."""
    base_msvd = ImplicitMSVD(factors=8, regularization=0.05, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    conf = sp.csr_matrix([[4.0, 0, 2.0], [0, 5.0, 0]], dtype=np.float32)
    base_msvd.fit(pref, conf, compute_loss_history=False, verbose=False)

    temp_mat = sp.csr_matrix([[0.5, 0.0, 0.866], [0.0, 1.0, 0.0]], dtype=np.float32)
    model = TemporalMSVD(base_msvd, temp_mat, alpha=0.2)

    assert model.n_users == 2
    assert model.n_items == 3
    assert model.alpha == 0.2


def test_alpha_zero_reproduces_base_msvd():
    """Verify alpha=0 yields identical ranking to pure base MSVD."""
    base_msvd = ImplicitMSVD(factors=8, regularization=0.05, iterations=3, seed=42)
    pref = sp.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
    conf = sp.csr_matrix([[2.0, 3.0, 0], [0, 1.0, 4.0]], dtype=np.float32)
    base_msvd.fit(pref, conf, compute_loss_history=False, verbose=False)

    temp_mat = sp.csr_matrix([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    temp_model = TemporalMSVD(base_msvd, temp_mat, alpha=0.0)

    for u in range(2):
        base_recs = base_msvd.recommend(u, k=3, exclude_seen=False)
        temp_recs = temp_model.recommend(u, k=3, exclude_seen=False)
        assert len(base_recs) == len(temp_recs)
        for (b_item, b_score), (t_item, t_score) in zip(base_recs, temp_recs):
            assert b_item == t_item
            assert pytest.approx(b_score, rel=1e-5) == t_score


def test_temporal_weight_boosts_candidate():
    """Verify positive alpha increases scores of temporally active items."""
    base_msvd = ImplicitMSVD(factors=4, regularization=0.01, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 0, 0]], dtype=np.float32)
    conf = sp.csr_matrix([[1.0, 0, 0]], dtype=np.float32)
    base_msvd.fit(pref, conf, compute_loss_history=False, verbose=False)

    # Item 2 has high temporal affinity
    temp_mat = sp.csr_matrix([[0.0, 0.0, 1.0]], dtype=np.float32)
    model = TemporalMSVD(base_msvd, temp_mat, alpha=10.0)

    recs = model.recommend(0, k=3, exclude_seen=False)
    # Item 2 should now be ranked #1 due to large alpha boost
    assert recs[0][0] == 2


def test_temporal_joint_tag_fusion():
    """Verify multi-stream MSVD + Tag + Temporal recommendation."""
    base_msvd = ImplicitMSVD(factors=4, regularization=0.01, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 0, 0]], dtype=np.float32)
    conf = sp.csr_matrix([[2.0, 0, 0]], dtype=np.float32)
    base_msvd.fit(pref, conf, compute_loss_history=False, verbose=False)

    tag_mat = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    temp_mat = sp.csr_matrix([[0.0, 1.0, 0.0]], dtype=np.float32)

    model = TemporalMSVD(base_msvd, temp_mat, alpha=0.5, tag_matrix=tag_mat, beta=0.5)
    model.fit_user_tag_profiles(pref, conf)

    recs = model.recommend(0, k=3, exclude_seen=True, seen_items={0})
    assert len(recs) == 2
    assert 0 not in [item for item, _ in recs]


def test_seen_items_excluded():
    """Verify seen items are strictly excluded from recommendations."""
    base_msvd = ImplicitMSVD(factors=4, regularization=0.01, iterations=2, seed=42)
    pref = sp.csr_matrix([[1, 1, 0]], dtype=np.float32)
    conf = sp.csr_matrix([[1.0, 1.0, 0]], dtype=np.float32)
    base_msvd.fit(pref, conf, compute_loss_history=False, verbose=False)

    temp_mat = sp.csr_matrix([[0.5, 0.5, 0.707]], dtype=np.float32)
    model = TemporalMSVD(base_msvd, temp_mat, alpha=1.0)

    recs = model.recommend(0, k=2, exclude_seen=True, seen_items={0, 1})
    rec_items = [item for item, _ in recs]
    assert 0 not in rec_items
    assert 1 not in rec_items
    assert 2 in rec_items
