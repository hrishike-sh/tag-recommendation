import math
from pathlib import Path
import duckdb
import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.als import ImplicitMSVD
from lastfm.baseline import PopularityRecommender
from lastfm.evaluate import evaluate_model, split_events
from lastfm.ingest import ingest
from lastfm.matrix import matrix
from lastfm.metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_popularity_recommender():
    # 3 users, 4 items
    # Training counts:
    # item 0: 10 plays
    # item 1: 25 plays
    # item 2: 5 plays
    # item 3: 30 plays
    # Sorted popularity: [3, 1, 0, 2]
    counts = sp.csr_matrix([
        [5, 10, 0, 15],
        [5, 15, 0, 15],
        [0, 0, 5, 0],
    ])
    pop = PopularityRecommender().fit(counts)
    assert list(pop.popular_item_indices) == [3, 1, 0, 2]

    # For user 0, seen items are {0, 1, 3}. Recommending with exclude_seen should yield item 2
    recs_u0 = pop.recommend(0, k=2, exclude_seen=True, seen_items={0, 1, 3})
    assert recs_u0 == [(2, 5.0)]


def test_synthetic_chronological_evaluation(tmp_path):
    """Synthetic dataset with known chronological split:

    Train cutoff: 2009-05-01T00:00:00Z
    Train events (< cutoff):
      u1: item A (count 10), item B (count 5)
      u2: item B (count 20), item C (count 2)
    Test events (>= cutoff):
      u1: item C (future positive)
      u2: item A (future positive)
    """
    tsv = tmp_path / "events.tsv"
    rows = [
        "1111111111111111111111111111111111111111\t2009-01-01T00:00:00Z\tart1\tA1\ttraA\tItemA",
        "1111111111111111111111111111111111111111\t2009-02-01T00:00:00Z\tart1\tA1\ttraB\tItemB",
        "2222222222222222222222222222222222222222\t2009-03-01T00:00:00Z\tart1\tA1\ttraB\tItemB",
        "2222222222222222222222222222222222222222\t2009-04-01T00:00:00Z\tart1\tA1\ttraC\tItemC",
        # Test period
        "1111111111111111111111111111111111111111\t2009-05-10T00:00:00Z\tart1\tA1\ttraC\tItemC",
        "2222222222222222222222222222222222222222\t2009-05-15T00:00:00Z\tart1\tA1\ttraA\tItemA",
    ]
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ingest(tmp_path, tsv, "1k")

    split = split_events(tmp_path, dataset="1k", train_cutoff="2009-05-01T00:00:00Z")
    assert split["n_users"] == 2
    assert split["n_items"] == 3
    assert len(split["test_positives_by_user"]) == 2

    # Fit Popularity model
    pop = PopularityRecommender().fit(split["Counts_train"])
    macro, per_user = evaluate_model(pop, split["P_train"], split["test_positives_by_user"], k_list=[1, 2])

    assert len(per_user) == 2
    for u in per_user:
        assert u["user_index"] in {0, 1}
        assert u["test_positives_count"] == 1
