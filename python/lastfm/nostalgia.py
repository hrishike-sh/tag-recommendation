"""Proposal Eq. 9 and two-stream discovery/rediscovery ranking.

This module does not change the locked Phase 9 three-signal experiments.
"""
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from .ingest import connect


def utc_cutoff(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('cutoff must include a timezone')
    return value.astimezone(timezone.utc)


def validate_parameters(recent_days, threshold, half_life_days):
    if not isinstance(recent_days, int) or recent_days <= 0:
        raise ValueError('recent_days must be a positive integer')
    if not isinstance(threshold, int) or threshold < 1:
        raise ValueError('threshold must be a positive integer')
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError('half_life_days must be positive and finite')


def listening_history(root: Path, user_id: str, cutoff: str, recent_days=90):
    """Per-user counts; active window includes its lower bound, excludes cutoff."""
    validate_parameters(recent_days, 1, 1)
    end = utc_cutoff(cutoff)
    boundary = end - timedelta(days=recent_days)
    with connect(root) as db:
        rows = db.execute('''
            SELECT item_id, count(*) FILTER (WHERE played_at < ?) AS historical_plays,
                   count(*) FILTER (WHERE played_at >= ?) AS active_plays,
                   min(played_at) AS first_played_at, max(played_at) AS last_played_at
            FROM recsys.events_1k WHERE user_id = ? AND played_at < ?
            GROUP BY item_id ORDER BY item_id
        ''', [boundary, boundary, user_id, end]).fetchall()
    return [dict(zip(('item_id', 'historical_plays', 'active_plays',
                      'first_played_at', 'last_played_at'), row)) for row in rows]


def nostalgia_score(plays, days_dormant, tag_similarity, half_life_days=90):
    """Eq. 9: log1p(C) * (1-exp(-lambda*delta_t)) * SimTag.

    Delta_t is elapsed days since last play; lambda = ln(2)/half_life_days.
    """
    validate_parameters(1, 1, half_life_days)
    if any(not math.isfinite(v) or v < 0 for v in (plays, days_dormant, tag_similarity)):
        raise ValueError('score inputs must be nonnegative and finite')
    if tag_similarity > 1:
        raise ValueError('tag similarity must be at most one')
    return math.log1p(plays) * -math.expm1(-math.log(2) * days_dormant / half_life_days) * tag_similarity


def score_dormant(history, item_ids, tags, cutoff, recent_days=90, threshold=5,
                  half_life_days=90):
    """Return every eligible dormant track, including zero-context-score tracks.

    SimTag is proposal Eq. 7: present-tag Jaccard times mean candidate
    L1 relevance on shared tags; context is the union of ACTIVE items' tags.
    Existing TF-IDF rows may be L2 normalized: re-normalize to L1 relevance.
    """
    validate_parameters(recent_days, threshold, half_life_days)
    end = utc_cutoff(cutoff)
    boundary = end - timedelta(days=recent_days)
    tags = sp.csr_matrix(tags, dtype=float, copy=True)
    tags.sum_duplicates(); tags.eliminate_zeros(); tags.sort_indices()
    if tags.shape[0] != len(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError('tag rows must align with unique item IDs')
    if not np.all(np.isfinite(tags.data)) or np.any(tags.data < 0):
        raise ValueError('tag relevance must be nonnegative and finite')
    mapping = {item: i for i, item in enumerate(item_ids)}
    context = set()
    for row in history:
        if row['active_plays'] > 0 and row['item_id'] in mapping:
            context.update(tags.getrow(mapping[row['item_id']]).indices.tolist())
    candidates = {}
    for row in history:
        idx = mapping.get(row['item_id'])
        if (idx is None or row['historical_plays'] < threshold or row['active_plays'] != 0
                or row['last_played_at'] >= boundary):
            continue
        tagrow = tags.getrow(idx)
        present = set(tagrow.indices.tolist())
        shared = sorted(present & context)
        relevance = dict(zip(tagrow.indices, tagrow.data / tagrow.data.sum())) if tagrow.nnz else {}
        similarity = (len(shared) / len(present | context) *
                      sum(relevance[t] for t in shared) / len(shared)) if shared else 0.0
        days = (end - row['last_played_at']).total_seconds() / 86400
        memory = math.log1p(row['historical_plays'])
        recovery = -math.expm1(-math.log(2) * days / half_life_days)
        candidates[idx] = dict(row, days_dormant=days, log_plays=memory,
            recovery=recovery, tag_similarity=similarity, shared_tag_indices=shared,
            score=nostalgia_score(row['historical_plays'], days, similarity, half_life_days))
    return candidates


class ExplanationTuple(NamedTuple):
    """Stable JSON array order, with field names also included in API response."""
    kind: str
    item_id: str
    stream_a: float
    stream_b: float
    alpha: float
    contribution_a: float
    contribution_b: float
    historical_plays: int
    active_plays: int
    days_dormant: float | None
    shared_tags: tuple


class DynamicTemporalArbiter:
    """alpha*A + (1-alpha)*B over unseen items union eligible dormant items.

    Automatic alpha = .2 + .6 * fraction of active items first heard in window.
    Missing positive streams fall back to the available stream in automatic mode.
    Explicit alpha is never silently overridden, including endpoints 0 and 1.
    """
    def recommend(self, stream_a, item_ids, history, dormant, cutoff, recent_days=90,
                  alpha=None, k=10, tag_names=None):
        if alpha is not None and (not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError('alpha must be finite and in [0,1]')
        if not isinstance(k, int) or k < 0:
            raise ValueError('k must be a nonnegative integer')
        validate_parameters(recent_days, 1, 1)
        end = utc_cutoff(cutoff)
        raw = np.asarray(stream_a, dtype=float)
        if raw.shape != (len(item_ids),) or not np.all(np.isfinite(raw)):
            raise ValueError('Stream A must be a finite catalog-aligned score vector')
        seen = {row['item_id'] for row in history}
        unseen = np.array([item not in seen for item in item_ids], dtype=bool)
        a = np.zeros(len(item_ids)); b = np.zeros(len(item_ids))
        # Nonnegative max scaling retains constant-positive streams and zero evidence.
        a[unseen] = np.maximum(raw[unseen], 0)
        for idx, row in dormant.items():
            b[idx] = row['score']
        a_scale = float(a.max(initial=0))
        b_scale = float(b.max(initial=0))
        if a_scale > 0: a /= a_scale
        if b_scale > 0: b /= b_scale
        active = [r for r in history if r['active_plays'] > 0]
        boundary = end - timedelta(days=recent_days)
        discovery_fraction = (sum(r['first_played_at'] >= boundary for r in active) / len(active)) if active else 0.5
        automatic = alpha is None
        if automatic:
            alpha = 0.2 + 0.6 * discovery_fraction
            if not np.any(b > 0): alpha = 1.0
            elif not np.any(a > 0): alpha = 0.0
        final = alpha * a + (1 - alpha) * b
        eligible = np.flatnonzero(final > 0)
        # Deterministic tie breaking by catalog index; no duplicate stream entries.
        if 0 < k < len(eligible):
            threshold_score = np.partition(final[eligible], -k)[-k]
            above = eligible[final[eligible] > threshold_score]
            tied = eligible[final[eligible] == threshold_score][:k-len(above)]
            eligible = np.concatenate((above, tied))
        selected = sorted(eligible, key=lambda i: (-final[i], int(i)))[:k]
        records = []
        for idx in selected:
            old = dormant.get(int(idx))
            names = tuple((tag_names or {}).get(t, f'tag:{t}') for t in old['shared_tag_indices']) if old else ()
            explanation = ExplanationTuple('nostalgia' if old else 'novelty', item_ids[idx],
                float(a[idx]), float(b[idx]), float(alpha), float(alpha*a[idx]),
                float((1-alpha)*b[idx]), old['historical_plays'] if old else 0,
                old['active_plays'] if old else 0, old['days_dormant'] if old else None, names)
            reason = (f"Resurfaced favorite: {old['historical_plays']} historical plays; "
                      f"{old['days_dormant']:.1f} days since last play; active-context tags: {', '.join(names)}.") if old else 'Unheard track ranked by collaborative affinity.'
            records.append(dict(item_id=item_ids[idx], score=float(final[idx]), kind=explanation.kind,
                reason=reason, explanation_tuple=explanation,
                stream_contributions={'a': explanation.contribution_a, 'b': explanation.contribution_b},
                nostalgia_evidence=({key: value.isoformat() if isinstance(value, datetime) else value
                                     for key, value in old.items()} if old else None)))
        return dict(alpha=float(alpha), alpha_mode='dynamic' if automatic else 'manual',
                    active_discovery_fraction=discovery_fraction,
                    score_scales={'a': a_scale, 'b': b_scale},
                    explanation_fields=list(ExplanationTuple._fields), recommendations=records)
