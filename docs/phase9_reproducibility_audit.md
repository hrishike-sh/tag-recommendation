# Phase 9: Reproducibility and Leakage Audit

**Audit Date:** 2026-09-24  
**Dataset Target:** Full-Scale Last.fm 1K + HetRec 2011 Artist Tags  
**Status:** **PASSED / ZERO LEAKAGE VERIFIED**

---

## 1. Dataset Snapshot & Integrity Identity

The canonical Last.fm 1K dataset stored in DuckDB (`data/lastfm.duckdb`) and Parquet lake was audited across all interaction rows, unique users, items, and timestamps.

| Entity | Canonical Audit Count | Phase 8 Baseline Count | Status |
| :--- | :--- | :--- | :--- |
| **Total Clean Events** | 19,150,865 | 19,150,865 | **Bitwise Identical** |
| **Total Users ($U$)** | 992 | 992 | **Bitwise Identical** |
| **Total Unique Tracks** | 1,505,185 | 1,505,185 | **Bitwise Identical** |
| **Test Catalog Items ($I$)** | 1,493,688 | 1,493,688 | **Bitwise Identical** |
| **HetRec Artist Tags ($T$)** | 2,299 | 2,299 | **Bitwise Identical** |
| **Tag Matrix NNZ** | 11,920,822 | 11,920,822 | **Bitwise Identical** |
| **Temporal Matrix NNZ** | 4,009,238 | 4,009,238 | **Bitwise Identical** |
| **Test Evaluable Users** | 518 | 518 | **Bitwise Identical** |

---

## 2. Chronological Boundary & Temporal Leakage Assertions

Strict timestamp partitioning was verified using exact timestamp bounds:

$$\max(T_{\text{train}}) < \min(T_{\text{val}}) < \max(T_{\text{val}}) < \min(T_{\text{test}})$$

```json
{
  "train_max_timestamp": "2009-03-31 23:59:57+00:00",
  "val_min_timestamp":   "2009-04-01 00:00:03+00:00",
  "val_max_timestamp":   "2009-04-30 23:59:57+00:00",
  "test_min_timestamp":  "2009-05-01 00:00:00+00:00",
  "test_max_timestamp":  "2009-06-19 21:31:16+00:00"
}
```

### Verified Pipeline Assertions:
1. **No Future Timestamps in Training:** All training interaction counts and preference confidence deltas contain zero events occurring at or after `2009-04-01T00:00:00Z` (for validation) and `2009-05-01T00:00:00Z` (for test).
2. **Tag Vocabulary Isolation:** Tag dictionary and document frequency calculations are derived strictly from training/catalog item identifiers and HetRec metadata without observing future test-period interactions.
3. **Temporal Profiles Causal Filtering:** User temporal affinity decays are calculated strictly with reference timestamp $t_{\text{ref}} = T_{\text{cutoff}}$ using only interactions prior to cutoff.
4. **Validation Isolation for Hyperparameters:** Normalization methods (`minmax`) and stream weights ($w = [0.20, 0.30, 0.50]$) were tuned strictly on validation interactions and locked prior to test evaluation.

---

## 3. Determinism Verification (Seed 42)

Three consecutive, independent test evaluation passes were executed on the Global Normalized Arbiter using seed 42.

| Metric | Run 1 | Run 2 | Run 3 | Max Absolute Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 0.0004698 | 0.0004698 | 0.0004698 | **0.0000000** |
| **Precision@5** | 0.0065637 | 0.0065637 | 0.0065637 | **0.0000000** |
| **NDCG@5** | 0.0072492 | 0.0072492 | 0.0072492 | **0.0000000** |
| **MAP@5** | 0.0048005 | 0.0048005 | 0.0048005 | **0.0000000** |
| **Recall@10** | 0.0007953 | 0.0007953 | 0.0007953 | **0.0000000** |
| **Precision@10** | 0.0071429 | 0.0071429 | 0.0071429 | **0.0000000** |
| **NDCG@10** | 0.0074142 | 0.0074142 | 0.0074142 | **0.0000000** |
| **MAP@10** | 0.0035540 | 0.0035540 | 0.0035540 | **0.0000000** |
| **Recall@20** | 0.0011971 | 0.0011971 | 0.0011971 | **0.0000000** |
| **Precision@20** | 0.0065637 | 0.0065637 | 0.0065637 | **0.0000000** |
| **NDCG@20** | 0.0069851 | 0.0069851 | 0.0069851 | **0.0000000** |
| **MAP@20** | 0.0024362 | 0.0024362 | 0.0024362 | **0.0000000** |

**Conclusion:** The recommendation pipeline is **bitwise deterministic** and free of numeric drift.
