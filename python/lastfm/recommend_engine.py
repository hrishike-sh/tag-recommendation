"""Production Recommendation Engine and Explainability Module for Multi-Stream Arbiter."""

from __future__ import annotations

import logging
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD
from .arbiter import MultiStreamArbiter, normalize_scores
from .split import split_train_val_test
from .tag_msvd import build_tag_matrix
from .temporal import compute_temporal_affinity_matrix

logger = logging.getLogger(__name__)


def get_recommendations_with_explanations(
    root: Path,
    user_query: str,
    k: int = 10,
    weights: Sequence[float] = (0.20, 0.30, 0.50),
    mode: str = "dual",
    alpha: Optional[float] = None,
    recent_days: int = 90,
    min_historical_plays: int = 5,
    half_life_days: float = 90,
) -> Dict[str, Any]:
    """Generate recommendations with stream contribution reasons for a given user.

    Parameters
    ----------
    root : Path
        Project root directory.
    user_query : str
        User identifier (either string user_id like 'user_000001' or integer index).
    k : int
        Number of recommendations to return.
    weights : Sequence[float]
        [w_msvd, w_tag, w_temporal] weights.
    """
    from .nostalgia import (listening_history, score_dormant, DynamicTemporalArbiter,
                            validate_parameters)
    if mode not in {"dual", "global"}:
        raise ValueError("mode must be dual or global")
    validate_parameters(recent_days, min_historical_plays, half_life_days)
    if alpha is not None and (not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must be finite and in [0,1]")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if mode == "global" and alpha is not None:
        raise ValueError("alpha is only available in dual mode")
    if mode == "dual":
        from .service import RecommendationService
        return RecommendationService.open(root).recommend(user_query, k=k, alpha=alpha,
            recent_days=recent_days, min_historical_plays=min_historical_plays,
            half_life_days=half_life_days)
    cutoff = "2009-05-01T00:00:00Z"
    # 1. Load splits and matrices
    splits = split_train_val_test(
        root,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        test_end="2009-07-01T00:00:00Z",
        kappa=40.0,
    )
    test_split = splits["test_split"]
    test_user_ids = test_split["user_ids"]
    test_item_ids = test_split["item_ids"]
    P_train = test_split["P_train"]
    D_train = test_split["D_train"]

    # Resolve user index
    if user_query.isdigit() and int(user_query) < len(test_user_ids):
        user_index = int(user_query)
        user_id = test_user_ids[user_index]
    elif user_query in test_user_ids:
        user_id = user_query
        user_index = test_user_ids.index(user_id)
    else:
        raise ValueError(f"Unknown user: {user_query}")

    ingest_report = json.loads((root / "artifacts/reports/ingest-1k.json").read_text())
    meta_parquet = Path(ingest_report["snapshot"]) / "track_metadata.parquet"
    if not meta_parquet.is_absolute():
        meta_parquet = root / meta_parquet

    # Fit MSVD model
    msvd = ImplicitMSVD(factors=64, regularization=0.05, iterations=10, seed=42)
    msvd.fit(P_train, D_train, compute_loss_history=False, verbose=False)

    tag_file = root / "data/raw/hetrec2011-lastfm-2k/user_taggedartists.dat"
    tags_available = tag_file.exists()
    if tags_available:
        T_test, vocabulary = build_tag_matrix(root, test_item_ids, meta_parquet,
            min_tag_freq=5, cutoff_iso=cutoff if mode == "dual" else None)
    else:
        T_test = sp.csr_matrix((len(test_item_ids), 0), dtype=np.float32)
        vocabulary = {}

    S_test = compute_temporal_affinity_matrix(
        root, user_ids=test_user_ids, item_ids=test_item_ids, cutoff_iso="2009-05-01T00:00:00Z", gamma=0.1, kappa=40.0
    )

    arbiter = MultiStreamArbiter(
        base_msvd=msvd,
        tag_matrix=T_test,
        temporal_matrix=S_test,
        weights=weights,
        normalization="minmax",
        gating_mode="global",
    )
    arbiter.fit_user_profiles(P_train, D_train)

    # Exclude seen items
    start = P_train.indptr[user_index]
    end = P_train.indptr[user_index + 1]
    seen_items = set(P_train.indices[start:end])

    # Compute raw stream components for explanation
    w = arbiter.get_user_weights(user_index)
    w_msvd, w_tag, w_temp = float(w[0]), float(w[1]), float(w[2])

    u_vec = msvd.user_factors[user_index]
    s_msvd_raw = u_vec @ msvd.item_factors.T
    s_msvd = normalize_scores(s_msvd_raw, method=arbiter.normalization)

    z_u = arbiter.user_tag_profiles[user_index] if arbiter.user_tag_profiles is not None else None
    s_tag = (T_test @ z_u) if (z_u is not None and T_test is not None) else np.zeros(len(test_item_ids), dtype=np.float32)

    s_temp = np.zeros(len(test_item_ids), dtype=np.float32)
    if S_test is not None and user_index < S_test.shape[0]:
        t_start = S_test.indptr[user_index]
        t_end = S_test.indptr[user_index + 1]
        if t_end > t_start:
            s_temp[S_test.indices[t_start:t_end]] = S_test.data[t_start:t_end]

    # Combined score
    combined = (w_msvd * s_msvd) + (w_tag * s_tag) + (w_temp * s_temp)
    for idx in seen_items:
        if 0 <= idx < len(test_item_ids):
            combined[idx] = -np.inf

    count = min(k, len(combined))
    top_indices = np.argpartition(combined, -count)[-count:] if count else np.array([], dtype=int)
    sorted_top = top_indices[np.argsort(-combined[top_indices])]

    # Fetch track metadata
    with duckdb.connect() as con:
        meta_rows = con.execute(
            f"SELECT item_id, artist_name, track_name FROM read_parquet('{meta_parquet}')"
        ).fetchall()

    meta_lookup = {r[0]: {"artist_name": r[1], "track_name": r[2]} for r in meta_rows}

    items_list = []
    for rank, idx in enumerate(sorted_top, start=1):
        raw_score = float(combined[idx])
        if np.isneginf(raw_score):
            continue
        
        c_msvd = float(w_msvd * s_msvd[idx])
        c_tag = float(w_tag * s_tag[idx])
        c_temp = float(w_temp * s_temp[idx])

        # Formulate human-readable explanation reason
        reasons = []
        contribs = [("Collaborative (MSVD)", c_msvd), ("Artist-Tag Affinity", c_tag), ("Temporal Recency", c_temp)]
        contribs.sort(key=lambda x: x[1], reverse=True)
        top_two = [name for name, val in contribs if val > 0.001][:2]
        explanation = " + ".join(top_two) if top_two else "Collaborative Baseline"

        item_id = test_item_ids[idx]
        meta = meta_lookup.get(item_id, {"artist_name": "Unknown Artist", "track_name": "Unknown Track"})

        items_list.append({
            "rank": rank,
            "item_id": item_id,
            "track_name": meta["track_name"],
            "artist_name": meta["artist_name"],
            "score": round(raw_score, 4),
            "reason": explanation,
            "stream_contributions": {
                "msvd": round(c_msvd, 4),
                "tag": round(c_tag, 4),
                "temporal": round(c_temp, 4),
            }
        })

    return {
        "user_index": user_index,
        "user_id": user_id,
        "model": "Global Normalized Arbiter",
        "weights": {"msvd": w_msvd, "tag": w_tag, "temporal": w_temp},
        "k": k,
        "recommendations": items_list,
    }
