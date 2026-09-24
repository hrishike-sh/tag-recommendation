"""Standard evaluation metrics for Top-K implicit recommendation."""

from __future__ import annotations

import math
from typing import Sequence, Set, Union


def recall_at_k(recommended: Sequence[int], test_positives: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Recall@K.
    
    Recall@K = |Recommended[:K] ∩ TestPositives| / |TestPositives|
    """
    if k <= 0:
        raise ValueError("k must be positive")
    positives = set(test_positives)
    if not positives:
        return 0.0
    rec_k = set(recommended[:k])
    hits = len(rec_k & positives)
    return hits / float(len(positives))


def precision_at_k(recommended: Sequence[int], test_positives: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Precision@K.
    
    Precision@K = |Recommended[:K] ∩ TestPositives| / K
    """
    if k <= 0:
        raise ValueError("k must be positive")
    positives = set(test_positives)
    if not positives:
        return 0.0
    rec_k = set(recommended[:k])
    hits = len(rec_k & positives)
    return hits / float(k)


def ndcg_at_k(recommended: Sequence[int], test_positives: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute NDCG@K using binary relevance.
    
    DCG@K = \\sum_{j=1}^K rel_j / log2(j + 1)
    IDCG@K = \\sum_{j=1}^{min(K, |TestPositives|)} 1 / log2(j + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if k <= 0:
        raise ValueError("k must be positive")
    positives = set(test_positives)
    if not positives:
        return 0.0

    # DCG
    dcg = 0.0
    rec_k = recommended[:k]
    for j, item in enumerate(rec_k, start=1):
        if item in positives:
            dcg += 1.0 / math.log2(j + 1)

    # IDCG
    num_ideal = min(k, len(positives))
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, num_ideal + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: Sequence[int], test_positives: Union[Set[int], Sequence[int]], k: int) -> float:
    """Compute Average Precision at K (MAP component for one user).
    
    AP@K = (1 / min(K, |TestPositives|)) * \\sum_{j=1}^K rel_j * Precision@j
    """
    if k <= 0:
        raise ValueError("k must be positive")
    positives = set(test_positives)
    if not positives:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    rec_k = recommended[:k]

    for j, item in enumerate(rec_k, start=1):
        if item in positives:
            hits += 1
            sum_precisions += hits / float(j)

    num_ideal = min(k, len(positives))
    if num_ideal == 0:
        return 0.0
    return sum_precisions / float(num_ideal)
