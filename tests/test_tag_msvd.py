import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.tag_msvd import TagFusedMSVD


def test_tag_fused_msvd_beta_zero():
    """Verify beta=0 gives identical scores to pure MSVD."""
    n_users, n_items, n_tags = 3, 4, 5
    P = sp.csr_matrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.float32)
    D = sp.csr_matrix([[10, 5, 0, 0], [0, 8, 2, 0], [0, 0, 4, 15]], dtype=np.float32)

    base_msvd = ImplicitMSVD(factors=4, regularization=0.05, iterations=5, seed=42)
    base_msvd.fit(P, D, verbose=False)

    # Random tag matrix
    T = sp.csr_matrix(np.random.default_rng(42).random((n_items, n_tags)), dtype=np.float32)

    tag_model = TagFusedMSVD(base_msvd, T, beta=0.0)
    tag_model.fit_user_profiles(P, D)

    # Compare recommendations
    recs_base = base_msvd.recommend(0, k=3, exclude_seen=False)
    recs_tag = tag_model.recommend(0, k=3, exclude_seen=False)

    for (i1, s1), (i2, s2) in zip(recs_base, recs_tag):
        assert i1 == i2
        assert s1 == pytest.approx(s2, rel=1e-5)


def test_tag_fused_msvd_boosting():
    """Verify that a positive tag similarity boosts candidate ranking."""
    n_users, n_items, n_tags = 2, 3, 2
    P = sp.csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    D = sp.csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    base_msvd = ImplicitMSVD(factors=2, regularization=0.05, iterations=5, seed=42)
    base_msvd.fit(P, D, verbose=False)

    # Item 0 and 2 share Tag 0; Item 1 has Tag 1
    T = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    tag_model = TagFusedMSVD(base_msvd, T, beta=2.0)
    tag_model.fit_user_profiles(P, D)

    # For user 0 (who listened to Item 0 with Tag 0), Item 2 should have high tag sim
    recs = tag_model.recommend(0, k=2, exclude_seen=True, seen_items={0})
    # Among unseen items {1, 2}, item 2 has tag similarity
    assert recs[0][0] == 2
