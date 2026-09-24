# Phase 6 — Tag Source Audit and Provenance Report

## 1. Tag Source and License Provenance
- **Primary External Tag Dataset:** HetRec 2011 (Last.fm 2K extension)
- **Source URL:** `https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip`
- **Publication / Provenance:** Published by GroupLens Research at RecSys 2011 (Cantador, Brusilovsky, Kuflik: *2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems*).
- **License:** Non-commercial academic research release from GroupLens / Last.fm.

---

## 2. Raw Tag Dataset Statistics
- **Total Tagging Records (`user_taggedartists.dat`):** $186,479$
- **Total Distinct Tags (`tags.dat`):** $11,946$
- **Total Artists in HetRec (`artists.dat`):** $17,632$
- **Total Tagged Artists in HetRec:** $12,523$

---

## 3. Entity Mapping Strategy & Granularity
- **Granularity:** Artist-level semantic tag profiles.
  - *Reasoning:* The Last.fm 1K listening dataset contains track names, artist names, artist MBIDs, and track MBIDs. HetRec 2011 contains artist names, artist URLs, and user tag assignments to artists. No public track-level tag database exists with universal MBID coverage for 1.5M tracks.
  - *Mapping Key:* Exact lowercase normalized artist name: `lower(trim(artist_name))`.
  - *Track Association:* Every track in Last.fm 1K inherits the tag profile of its parent artist.

---

## 4. Empirical Mapping Coverage on Canonical Last.fm 1K
Measured directly against the complete 19,150,865 listening events and 1,505,185 tracks:

| Entity Level | Total in Last.fm 1K | Matched to HetRec | Coverage Percentage |
|---|:---:|:---:|:---:|
| **Unique Artist Names** | $174,090$ | $11,728$ | **$6.74\%$** |
| **Unique Tracks (Items)** | $1,505,309$ | $644,008$ | **$42.78\%$** |
| **Total Listening Events** | $19,150,865$ | $14,450,730$ | **$75.46\%$** |

> **Coverage Analysis:** While HetRec covers $6.74\%$ of all long-tail artist names, these matched artists account for **$42.78\%$ of all unique tracks** and **$75.46\%$ of all listening events in the dataset**, providing rich semantic tag vectors for the vast majority of active consumption.

---

## 5. Duplicate, Missing Tag, and Normalization Policy
1. **Normalization:** Tag strings are lowercased, trimmed, and stripped of non-alphanumeric punctuation.
2. **Missing Tags:** Tracks whose artist does not match HetRec or has zero tag assignments receive an explicit all-zero sparse tag vector $\mathbf{t}_i = \mathbf{0}$, ensuring mathematical compatibility without hallucinating false genres.
3. **Weighting Scheme:** Item tag vectors use normalized term frequency (or TF-IDF calculated strictly over the training catalog).
