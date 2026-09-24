"""Chronological split and hyperparameter search on validation period."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD
from .baseline import PopularityRecommender
from .evaluate import evaluate_model
from .ingest import connect, literal


def split_train_val_test(
    root: Path,
    dataset: str = "1k",
    train_cutoff: str = "2009-04-01T00:00:00Z",
    val_cutoff: str = "2009-05-01T00:00:00Z",
    test_end: Optional[str] = "2009-07-01T00:00:00Z",
    kappa: float = 40.0,
) -> Dict[str, Any]:
    """Chronologically split events into:

    1. TRAIN: [< 2009-04-01)
    2. VALIDATION: [2009-04-01, 2009-05-01)
    3. TEST: [2009-05-01, test_end)
    Also builds TRAIN+VAL for the final test model fit: [< 2009-05-01).
    """
    if dataset != "1k":
        raise ValueError("Chronological split requires 1k dataset")

    t_train = datetime.fromisoformat(train_cutoff.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    t_val = datetime.fromisoformat(val_cutoff.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()

    with connect(root) as db:
        train_lit = f"TIMESTAMPTZ {literal(t_train)}"
        val_lit = f"TIMESTAMPTZ {literal(t_val)}"

        # --- A. TRAIN PERIOD (< train_cutoff) ---
        db.execute(f"""
            CREATE TEMP TABLE t_train_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE (year < year({train_lit}) OR (year=year({train_lit}) AND month<=month({train_lit})))
              AND played_at < {train_lit}
            GROUP BY 1, 2
        """)

        db.execute("""
            CREATE TEMP TABLE map_users_train AS
            SELECT user_id, (row_number() OVER (ORDER BY user_id)-1)::BIGINT AS user_index
            FROM (SELECT DISTINCT user_id FROM t_train_counts)
        """)

        db.execute("""
            CREATE TEMP TABLE map_items_train AS
            SELECT item_id, (row_number() OVER (ORDER BY item_id)-1)::BIGINT AS item_index
            FROM (SELECT DISTINCT item_id FROM t_train_counts)
        """)

        db.execute(f"""
            CREATE TEMP TABLE t_train_edges AS
            SELECT u.user_index, i.item_index, tc.play_count,
                   1::UTINYINT AS preference,
                   {kappa} * ln(1.0 + tc.play_count) AS confidence_delta
            FROM t_train_counts tc
            JOIN map_users_train u USING(user_id)
            JOIN map_items_train i USING(item_id)
        """)

        # --- B. VALIDATION PERIOD ([train_cutoff, val_cutoff)) ---
        db.execute(f"""
            CREATE TEMP TABLE t_val_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE played_at >= {train_lit} AND played_at < {val_lit}
            GROUP BY 1, 2
        """)

        db.execute("""
            CREATE TEMP TABLE t_val_edges AS
            SELECT u.user_index, i.item_index, vc.play_count
            FROM t_val_counts vc
            JOIN map_users_train u USING(user_id)
            JOIN map_items_train i USING(item_id)
        """)

        # --- C. TRAIN+VAL COMBINED FOR TEST FIT (< val_cutoff) ---
        db.execute(f"""
            CREATE TEMP TABLE t_trainval_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE (year < year({val_lit}) OR (year=year({val_lit}) AND month<=month({val_lit})))
              AND played_at < {val_lit}
            GROUP BY 1, 2
        """)

        db.execute("""
            CREATE TEMP TABLE map_users_trainval AS
            SELECT user_id, (row_number() OVER (ORDER BY user_id)-1)::BIGINT AS user_index
            FROM (SELECT DISTINCT user_id FROM t_trainval_counts)
        """)

        db.execute("""
            CREATE TEMP TABLE map_items_trainval AS
            SELECT item_id, (row_number() OVER (ORDER BY item_id)-1)::BIGINT AS item_index
            FROM (SELECT DISTINCT item_id FROM t_trainval_counts)
        """)

        db.execute(f"""
            CREATE TEMP TABLE t_trainval_edges AS
            SELECT u.user_index, i.item_index, tvc.play_count,
                   1::UTINYINT AS preference,
                   {kappa} * ln(1.0 + tvc.play_count) AS confidence_delta
            FROM t_trainval_counts tvc
            JOIN map_users_trainval u USING(user_id)
            JOIN map_items_trainval i USING(item_id)
        """)

        # --- D. TEST PERIOD (>= val_cutoff) ---
        test_where = f"played_at >= {val_lit}"
        if test_end:
            t_end = datetime.fromisoformat(test_end.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            test_where += f" AND played_at < TIMESTAMPTZ {literal(t_end)}"

        db.execute(f"""
            CREATE TEMP TABLE t_test_counts AS
            SELECT user_id, item_id, count(*)::BIGINT AS play_count
            FROM recsys.events_1k
            WHERE {test_where}
            GROUP BY 1, 2
        """)

        db.execute("""
            CREATE TEMP TABLE t_test_edges AS
            SELECT u.user_index, i.item_index, tc.play_count
            FROM t_test_counts tc
            JOIN map_users_trainval u USING(user_id)
            JOIN map_items_trainval i USING(item_id)
        """)

        # Fetch matrices
        train_data = db.execute("SELECT user_index, item_index, play_count, confidence_delta FROM t_train_edges").fetchnumpy()
        val_data = db.execute("SELECT user_index, item_index FROM t_val_edges").fetchnumpy()
        train_user_ids = [r[0] for r in db.execute("SELECT user_id FROM map_users_train ORDER BY user_index").fetchall()]
        train_item_ids = [r[0] for r in db.execute("SELECT item_id FROM map_items_train ORDER BY item_index").fetchall()]
        n_train_u = len(train_user_ids)
        n_train_i = len(train_item_ids)

        trainval_data = db.execute("SELECT user_index, item_index, play_count, confidence_delta FROM t_trainval_edges").fetchnumpy()
        test_data = db.execute("SELECT user_index, item_index FROM t_test_edges").fetchnumpy()
        trainval_user_ids = [r[0] for r in db.execute("SELECT user_id FROM map_users_trainval ORDER BY user_index").fetchall()]
        trainval_item_ids = [r[0] for r in db.execute("SELECT item_id FROM map_items_trainval ORDER BY item_index").fetchall()]
        n_trainval_u = len(trainval_user_ids)
        n_trainval_i = len(trainval_item_ids)

    # 1. Validation split matrices
    P_train = sp.csr_matrix(
        (np.ones(len(train_data["user_index"]), dtype=np.float32), (train_data["user_index"], train_data["item_index"])),
        shape=(n_train_u, n_train_i),
    )
    D_train = sp.csr_matrix(
        (train_data["confidence_delta"].astype(np.float32), (train_data["user_index"], train_data["item_index"])),
        shape=(n_train_u, n_train_i),
    )
    Counts_train = sp.csr_matrix(
        (train_data["play_count"].astype(np.float32), (train_data["user_index"], train_data["item_index"])),
        shape=(n_train_u, n_train_i),
    )

    val_positives: Dict[int, Set[int]] = {}
    for u_idx, i_idx in zip(val_data["user_index"], val_data["item_index"]):
        u, i = int(u_idx), int(i_idx)
        if u not in val_positives:
            val_positives[u] = set()
        val_positives[u].add(i)

    # 2. Test split matrices
    P_trainval = sp.csr_matrix(
        (np.ones(len(trainval_data["user_index"]), dtype=np.float32), (trainval_data["user_index"], trainval_data["item_index"])),
        shape=(n_trainval_u, n_trainval_i),
    )
    D_trainval = sp.csr_matrix(
        (trainval_data["confidence_delta"].astype(np.float32), (trainval_data["user_index"], trainval_data["item_index"])),
        shape=(n_trainval_u, n_trainval_i),
    )
    Counts_trainval = sp.csr_matrix(
        (trainval_data["play_count"].astype(np.float32), (trainval_data["user_index"], trainval_data["item_index"])),
        shape=(n_trainval_u, n_trainval_i),
    )

    test_positives: Dict[int, Set[int]] = {}
    for u_idx, i_idx in zip(test_data["user_index"], test_data["item_index"]):
        u, i = int(u_idx), int(i_idx)
        if u not in test_positives:
            test_positives[u] = set()
        test_positives[u].add(i)

    return {
        "train_cutoff": t_train,
        "val_cutoff": t_val,
        "test_end": test_end,
        "validation_split": {
            "P_train": P_train,
            "D_train": D_train,
            "Counts_train": Counts_train,
            "val_positives": val_positives,
            "user_ids": train_user_ids,
            "item_ids": train_item_ids,
            "n_users": n_train_u,
            "n_items": n_train_i,
            "evaluable_users": len(val_positives),
        },
        "test_split": {
            "P_train": P_trainval,
            "D_train": D_trainval,
            "Counts_train": Counts_trainval,
            "test_positives": test_positives,
            "user_ids": trainval_user_ids,
            "item_ids": trainval_item_ids,
            "n_users": n_trainval_u,
            "n_items": n_trainval_i,
            "evaluable_users": len(test_positives),
        },
    }
