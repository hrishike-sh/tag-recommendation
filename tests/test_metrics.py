import math
import pytest
from lastfm.metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_metric_manual_example():
    """Manual example from prompt:

    Test positives: {i2, i5}
    Recommendations: [i5, i7, i2]
    """
    positives = {2, 5}
    recs = [5, 7, 2]

    # Recall
    assert recall_at_k(recs, positives, k=1) == pytest.approx(1.0 / 2.0)  # {5} / 2
    assert recall_at_k(recs, positives, k=2) == pytest.approx(1.0 / 2.0)  # {5} / 2
    assert recall_at_k(recs, positives, k=3) == pytest.approx(2.0 / 2.0)  # {5, 2} / 2

    # Precision
    assert precision_at_k(recs, positives, k=1) == pytest.approx(1.0 / 1.0)
    assert precision_at_k(recs, positives, k=2) == pytest.approx(1.0 / 2.0)
    assert precision_at_k(recs, positives, k=3) == pytest.approx(2.0 / 3.0)

    # NDCG
    # At K=1: DCG = 1 / log2(2) = 1.0. IDCG = 1 / log2(2) = 1.0 => NDCG = 1.0
    assert ndcg_at_k(recs, positives, k=1) == pytest.approx(1.0)
    # At K=2: DCG = 1 / log2(2) + 0 = 1.0. IDCG = 1 / log2(2) + 1 / log2(3) => 1.0 / (1 + 1/log2(3))
    idcg_2 = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(recs, positives, k=2) == pytest.approx(1.0 / idcg_2)
    # At K=3: DCG = 1.0 + 0 + 1 / log2(4) = 1.0 + 0.5 = 1.5. IDCG = 1.0 + 1/log2(3)
    assert ndcg_at_k(recs, positives, k=3) == pytest.approx(1.5 / idcg_2)

    # MAP / AP
    # At K=1: hit at 1 (P@1 = 1). AP@1 = (1 * 1.0) / min(1, 2) = 1.0
    assert average_precision_at_k(recs, positives, k=1) == pytest.approx(1.0)
    # At K=2: hit at 1. sum_prec = 1.0. AP@2 = 1.0 / min(2, 2) = 0.5
    assert average_precision_at_k(recs, positives, k=2) == pytest.approx(0.5)
    # At K=3: hits at 1 and 3. P@1 = 1.0, P@3 = 2/3. sum_prec = 1 + 2/3 = 5/3. AP@3 = (5/3) / 2 = 5/6
    assert average_precision_at_k(recs, positives, k=3) == pytest.approx((5.0 / 3.0) / 2.0)


def test_metric_perfect_ranking():
    positives = {10, 20, 30}
    recs = [10, 20, 30, 40, 50]
    assert recall_at_k(recs, positives, k=3) == pytest.approx(1.0)
    assert precision_at_k(recs, positives, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(recs, positives, k=3) == pytest.approx(1.0)
    assert average_precision_at_k(recs, positives, k=3) == pytest.approx(1.0)


def test_metric_no_hits():
    positives = {1, 2}
    recs = [3, 4, 5]
    assert recall_at_k(recs, positives, k=3) == 0.0
    assert precision_at_k(recs, positives, k=3) == 0.0
    assert ndcg_at_k(recs, positives, k=3) == 0.0
    assert average_precision_at_k(recs, positives, k=3) == 0.0


def test_metric_one_hit():
    positives = {1}
    recs = [2, 1, 3]
    assert recall_at_k(recs, positives, k=2) == 1.0
    assert precision_at_k(recs, positives, k=2) == 0.5
    # DCG@2 = 1 / log2(3), IDCG@1 = 1 / log2(2) = 1.0 => NDCG@2 = 1 / log2(3)
    assert ndcg_at_k(recs, positives, k=2) == pytest.approx(1.0 / math.log2(3))
    # AP@2 = (1/2) / 1 = 0.5
    assert average_precision_at_k(recs, positives, k=2) == pytest.approx(0.5)


def test_metric_all_positives_beyond_k():
    positives = {9, 10}
    recs = [1, 2, 3, 4, 9, 10]
    assert recall_at_k(recs, positives, k=3) == 0.0
    assert precision_at_k(recs, positives, k=3) == 0.0
    assert ndcg_at_k(recs, positives, k=3) == 0.0
    assert average_precision_at_k(recs, positives, k=3) == 0.0


def test_metric_fewer_than_k_recommendations():
    positives = {1, 2}
    recs = [1]
    assert recall_at_k(recs, positives, k=5) == 0.5
    assert precision_at_k(recs, positives, k=5) == 0.2
    assert ndcg_at_k(recs, positives, k=5) == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))
    assert average_precision_at_k(recs, positives, k=5) == pytest.approx(1.0 / 2.0)


def test_metric_zero_positives():
    positives = set()
    recs = [1, 2, 3]
    assert recall_at_k(recs, positives, k=3) == 0.0
    assert precision_at_k(recs, positives, k=3) == 0.0
    assert ndcg_at_k(recs, positives, k=3) == 0.0
    assert average_precision_at_k(recs, positives, k=3) == 0.0


def test_metric_empty_recommendations():
    positives = {1, 2}
    recs = []
    assert recall_at_k(recs, positives, k=3) == 0.0
    assert precision_at_k(recs, positives, k=3) == 0.0
    assert ndcg_at_k(recs, positives, k=3) == 0.0
    assert average_precision_at_k(recs, positives, k=3) == 0.0
