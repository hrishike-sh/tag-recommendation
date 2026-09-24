import json
from pathlib import Path
import pytest
import scipy.sparse as sp

from lastfm.ingest import ingest
from lastfm.protocol import run_phase4_protocol
from lastfm.split import split_train_val_test


def test_split_train_val_test_isolation(tmp_path):
    tsv = tmp_path / "events.tsv"
    rows = [
        # Train (< 2009-04-01): User 0 has A, User 1 has B
        "1111111111111111111111111111111111111111\t2009-01-01T00:00:00Z\tart1\tA1\ttraA\tItemA",
        "2222222222222222222222222222222222222222\t2009-03-01T00:00:00Z\tart1\tA1\ttraB\tItemB",
        # Val ([2009-04-01, 2009-05-01)): User 0 listens to item B and new item C
        "1111111111111111111111111111111111111111\t2009-04-10T00:00:00Z\tart1\tA1\ttraB\tItemB",
        "1111111111111111111111111111111111111111\t2009-04-20T00:00:00Z\tart1\tA1\ttraC\tItemC",
        # Test (>= 2009-05-01): User 1 listens to item A
        "2222222222222222222222222222222222222222\t2009-05-15T00:00:00Z\tart1\tA1\ttraA\tItemA",
    ]
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ingest(tmp_path, tsv, "1k")

    splits = split_train_val_test(
        tmp_path,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        test_end="2009-07-01T00:00:00Z",
    )

    val_split = splits["validation_split"]
    test_split = splits["test_split"]

    # In validation split, train has 2 items (A, B). Item C was in validation period.
    assert val_split["n_users"] == 2
    assert val_split["n_items"] == 2
    assert len(val_split["val_positives"]) == 1  # only user 0 had item C

    # In test split, train has 3 items (A, B, C) because validation is included in train+val for final test.
    assert test_split["n_users"] == 2
    assert test_split["n_items"] == 3
    assert len(test_split["test_positives"]) == 1  # user 1 had item A


def test_phase4_protocol_execution(tmp_path):
    tsv = tmp_path / "events.tsv"
    rows = [
        # Train
        "1111111111111111111111111111111111111111\t2009-01-01T00:00:00Z\tart1\tA1\ttraA\tItemA",
        "1111111111111111111111111111111111111111\t2009-02-01T00:00:00Z\tart1\tA1\ttraB\tItemB",
        "2222222222222222222222222222222222222222\t2009-03-01T00:00:00Z\tart1\tA1\ttraB\tItemB",
        "2222222222222222222222222222222222222222\t2009-03-15T00:00:00Z\tart1\tA1\ttraC\tItemC",
        # Val
        "1111111111111111111111111111111111111111\t2009-04-10T00:00:00Z\tart1\tA1\ttraC\tItemC",
        "2222222222222222222222222222222222222222\t2009-04-12T00:00:00Z\tart1\tA1\ttraA\tItemA",
        # Test
        "1111111111111111111111111111111111111111\t2009-05-10T00:00:00Z\tart1\tA1\ttraC\tItemC",
        "2222222222222222222222222222222222222222\t2009-05-15T00:00:00Z\tart1\tA1\ttraA\tItemA",
    ]
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ingest(tmp_path, tsv, "1k")

    res = run_phase4_protocol(
        tmp_path,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        factors_grid=[4, 8],
        reg_grid=[0.05],
        iterations_grid=[5],
        seed=42,
        out_dir=tmp_path / "results",
    )

    assert "selected_config" in res
    assert "popularity_metrics" in res
    assert "als_metrics" in res
    assert (tmp_path / "results/validation/best_config.json").exists()
    assert (tmp_path / "results/test/final_results.json").exists()
