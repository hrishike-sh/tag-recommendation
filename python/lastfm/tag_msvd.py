"""Tag-fused MSVD (Stream A) implementation and HetRec pipeline."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD

logger = logging.getLogger(__name__)


def build_tag_matrix(
    root: Path,
    item_ids: Sequence[str],
    track_metadata_parquet: Path,
    min_tag_freq: int = 5,
    cutoff_iso: Optional[str] = None,
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """Build L2-normalized TF-IDF item tag matrix (I x T) for candidate items."""
    import duckdb
    from .ingest import literal

    raw_tags = root / "data/raw/hetrec2011-lastfm-2k"
    if not (raw_tags / "user_taggedartists.dat").exists():
        raise FileNotFoundError(f"Import HetRec data into {raw_tags} before building tags")

    from .nostalgia import utc_cutoff
    from datetime import datetime, timezone
    tag_cutoff = utc_cutoff(cutoff_iso) if cutoff_iso else None

    # 1. Load HetRec artists and tags
    hetrec_artists: Dict[int, str] = {}
    with open(raw_tags / "artists.dat", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 2:
                hetrec_artists[int(row[0])] = row[1].strip().lower()

    # Tag assignments: artist_name -> {tag_id: count}
    artist_tag_counts: Dict[str, Dict[int, int]] = {}
    tag_global_freq: Dict[int, int] = {}
    with open(raw_tags / "user_taggedartists.dat", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                if tag_cutoff is not None:
                    if len(row) < 6:
                        raise ValueError("Timestamped HetRec assignments required for historical tags")
                    assigned = datetime(int(row[5]), int(row[4]), int(row[3]), tzinfo=timezone.utc)
                    if assigned >= tag_cutoff:
                        continue
                a_id, t_id = int(row[1]), int(row[2])
                if a_id in hetrec_artists:
                    a_name = hetrec_artists[a_id]
                    if a_name not in artist_tag_counts:
                        artist_tag_counts[a_name] = {}
                    artist_tag_counts[a_name][t_id] = artist_tag_counts[a_name].get(t_id, 0) + 1
                    tag_global_freq[t_id] = tag_global_freq.get(t_id, 0) + 1

    # Filter tags by frequency
    filtered_tags = [t_id for t_id, freq in sorted(tag_global_freq.items()) if freq >= min_tag_freq]
    retained_tags = {t_id: idx for idx, t_id in enumerate(filtered_tags)}
    n_tags = len(retained_tags)

    # 2. Map item_ids to artist_name
    db = duckdb.connect()
    item_rows = db.execute(f"""
        SELECT item_id, lower(trim(artist_name)) AS artist_norm
        FROM read_parquet({literal(track_metadata_parquet.as_posix())})
        WHERE artist_name IS NOT NULL
        ORDER BY item_id, artist_norm
    """).fetchall()
    db.close()
    item_to_artist = {r[0]: r[1] for r in reversed(item_rows)}

    # 3. Construct sparse TF matrix (I x T)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    doc_freq = np.zeros(n_tags, dtype=np.float32)

    for i_idx, item_id in enumerate(item_ids):
        a_name = item_to_artist.get(item_id)
        if a_name and a_name in artist_tag_counts:
            t_dict = artist_tag_counts[a_name]
            for t_id, cnt in t_dict.items():
                if t_id in retained_tags:
                    t_idx = retained_tags[t_id]
                    rows.append(i_idx)
                    cols.append(t_idx)
                    vals.append(float(cnt))
                    doc_freq[t_idx] += 1.0

    n_items = len(item_ids)
    if not rows:
        T_mat = sp.csr_matrix((n_items, n_tags), dtype=np.float32)
        return T_mat, retained_tags

    TF_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_tags), dtype=np.float32)

    # Compute IDF strictly on item catalog
    idf = np.log((1.0 + n_items) / (1.0 + doc_freq)) + 1.0
    TFIDF_mat = TF_mat.multiply(idf).tocsr()

    # L2 normalize rows
    norms = np.sqrt(np.asarray(TFIDF_mat.power(2).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv_norms = 1.0 / norms
    T_mat = TFIDF_mat.multiply(inv_norms[:, None]).tocsr().astype(np.float32)

    return T_mat, retained_tags


class TagFusedMSVD:
    """Tag-Fused Implicit MSVD Recommender (Stream A)."""

    def __init__(
        self,
        base_msvd: ImplicitMSVD,
        tag_matrix: sp.csr_matrix,
        beta: float = 0.1,
    ) -> None:
        """Initialize Tag-Fused MSVD.

        Parameters
        ----------
        base_msvd : ImplicitMSVD
            Fitted base collaborative MSVD model.
        tag_matrix : sparse CSR matrix (n_items x n_tags)
            L2-normalized item tag representation.
        beta : float, default=0.1
            Semantic fusion coefficient.
        """
        self.base_msvd = base_msvd
        self.tag_matrix = tag_matrix.tocsr().astype(np.float32)
        self.beta = beta

        self.user_tag_profiles: Optional[np.ndarray] = None  # Shape: (n_users, n_tags)
        self.n_users = base_msvd.n_users
        self.n_items = base_msvd.n_items

    def fit_user_profiles(
        self,
        pref_csr: sp.csr_matrix,
        conf_delta_csr: sp.csr_matrix,
    ) -> "TagFusedMSVD":
        """Compute L2-normalized user semantic tag profiles z_u = \sum c_{ui} t_i."""
        n_users = pref_csr.shape[0]
        n_tags = self.tag_matrix.shape[1]
        profiles = np.zeros((n_users, n_tags), dtype=np.float32)

        # Vectorized aggregation: C * P @ T
        # Since c_ui * p_ui = (1 + d_ui) * p_ui
        c_sparse = pref_csr.copy().astype(np.float32)
        c_sparse.data = (1.0 + conf_delta_csr.data.astype(np.float32)) * pref_csr.data.astype(np.float32)

        raw_profiles = c_sparse @ self.tag_matrix  # Shape: (n_users, n_tags)
        profiles = raw_profiles.toarray() if sp.issparse(raw_profiles) else np.asarray(raw_profiles)

        # L2 normalize user profiles
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_tag_profiles = (profiles / norms).astype(np.float32)
        return self

    def recommend(
        self,
        user_index: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[Union[Sequence[int], Set[int], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommendations fusing collaborative MSVD and semantic tag similarity."""
        if self.base_msvd.user_factors is None or self.base_msvd.item_factors is None:
            raise RuntimeError("Base MSVD model is not fitted.")
        if self.user_tag_profiles is None:
            raise RuntimeError("User tag profiles have not been computed yet.")
        if k <= 0:
            return []

        # 1. Base MSVD scores: x_u @ Y^T (shape: n_items)
        u_vec = self.base_msvd.user_factors[user_index]
        scores = u_vec @ self.base_msvd.item_factors.T

        # 2. Semantic tag similarity: z_u @ T^T
        if self.beta > 0.0:
            z_u = self.user_tag_profiles[user_index]
            # Fast sparse dot product: tag_matrix @ z_u
            tag_sim = self.tag_matrix @ z_u  # Shape: (n_items,)
            scores = scores + (self.beta * tag_sim)

        if exclude_seen and seen_items is not None:
            scores = scores.copy()
            for idx in seen_items:
                if 0 <= idx < self.n_items:
                    scores[idx] = -np.inf

        num_candidates = min(k, self.n_items)
        if num_candidates == 0:
            return []

        if num_candidates < self.n_items:
            candidate_indices = np.argpartition(scores, -num_candidates)[-num_candidates:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: List[Tuple[int, float]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if np.isneginf(score):
                continue
            results.append((int(idx), score))
            if len(results) >= k:
                break

        return results
