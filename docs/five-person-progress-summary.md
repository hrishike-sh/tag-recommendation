# Five-person recorded progress summary — main-final

Prepared 24 September 2026. Based on the checked-in Phase 6–9 implementation and reports on main-final (6914116). Approximately 10–12 minutes. These labels divide speaking roles, not implementation credit. Evaluation numbers below are recorded results, not experiments rerun for this summary.

## Speaker 1 — Project objective and data foundation

Good morning, sir. Today we will summarize our music recommendation project and the progress represented by the main-final branch. Our objective is to combine listening behavior, descriptive music tags, and time-dependent preferences to produce personalized track recommendations.

We have divided this update into five parts. I will introduce the data and project structure. The second speaker will explain collaborative recommendation. The third speaker will cover tags. The fourth speaker will explain temporal preferences and the combined ranking system. The fifth speaker will present the evaluation findings, limitations, and next steps.

The foundation is the Last.fm 1K listening dataset. After cleaning, it contains approximately 19.15 million listening events from 992 users. These events include timestamps, so we can distinguish earlier listening from later listening. We also acquired the Last.fm 360K dataset, but that dataset contains aggregated artist play counts rather than timestamped track events. It remains a separate resource; the reported multi-stream experiments focus on 1K with additional HetRec tags.

The ingestion pipeline uses DuckDB and Parquet. The timestamped data is organized into year and month partitions. This provides a practical way to process the large listening history and build repeatable training snapshots without putting all data into ordinary Python lists.

An important part of our experiment is chronological evaluation. The documented protocol uses data before April 2009 for initial training, April for validation, and May through June for testing. Validation is used to choose settings, while the later test period measures performance after selection. The final test catalog contains roughly 1.49 million items. The protocol distinguishes the initial training stage from the later snapshot used for test evaluation.

Python contains the research pipeline and recommendation components. Go provides an initial scaffold and the shared confidence calculation. The repository also contains evaluation outputs and reports. The top-level README still describes the earlier data-foundation milestone, so the Phase 6–9 documents are the more relevant references for the current branch.

The main change since the early handover is that we now have implementations and recorded experiments covering collaborative, tag-based, temporal, and combined recommendation. I will hand over to the second speaker for the collaborative component.

## Speaker 2 — Implicit feedback and collaborative recommendation

Thank you. The collaborative component learns from listening behavior. We do not have explicit ratings for every user and track. Instead, we have implicit feedback: whether a person played a track and how many times they played it.

We separate preference from confidence. An observed play indicates a positive preference, and repeated plays increase confidence in that observation. The confidence formula is one plus forty times the natural logarithm of one plus the play count. This makes repeated listening meaningful while reducing the influence of extremely large counts.

An unplayed item does not necessarily mean a user dislikes it. It may simply mean that they have never encountered it. The learning objective therefore gives unobserved interactions lower confidence than strongly observed ones.

The implementation uses an implicit matrix-factorization model trained through alternating least squares. It learns a compact vector for each user and each item. The dot product between a user vector and an item vector becomes a collaborative recommendation score. The branch calls this component implicit MSVD; the underlying mechanism should be explained as confidence-weighted implicit factorization rather than as evidence that every formulation from the original proposal has been reproduced.

This method can discover relationships that are not obvious from artist names alone. If users with overlapping listening patterns also listen to another track, the model can give that track a higher score for a similar user. The factors are learned numerical patterns, not manually named genres.

The evaluation compares this model with a popularity baseline. Popularity gives broadly popular tracks high ranks without learning a separate preference vector for each person. That is a useful reference because a personalized system should demonstrate value beyond recommending the same popular music to everyone.

The pipeline uses sparse representations so that we do not need to store a complete dense user-by-track table. This is especially important when the catalog contains more than a million items. Recommendations are ranked for individual users, and the evaluation protocol handles exclusion of items already observed in the relevant history.

The collaborative model is our central signal, but it has limitations. Sparse histories provide less evidence, and numerical factors do not directly describe musical characteristics. The next speaker will explain how the tag component introduces descriptive information.

## Speaker 3 — Real tag data and semantic similarity

Thank you. My part concerns the semantic component, which uses tags to describe musical interests. This branch has progressed beyond the earlier implementation that only had synthetic tag fixtures: it includes a real tag pipeline based on the HetRec 2011 Last.fm dataset.

These tags are assigned to artists. We match artist names after normalization, and tracks inherit the tag profile of their matched artist. This is an important boundary: we are using artist-level descriptions as track features, not claiming that each track has an independently collected set of tags.

We represent these descriptions using TF-IDF. Term frequency reflects how often a tag is assigned to an artist, while inverse document frequency reduces the weight of tags that are common throughout the catalog. The resulting vectors are normalized so that their direction captures the tag mixture rather than simply the number of assignments.

A user's tag profile combines the tag vectors of tracks in their listening history, weighted by confidence. The system then compares that normalized profile with a candidate track's tag vector using cosine similarity. A strong match means that the candidate's artist tags resemble the tag patterns in the user's history.

This is different from the weighted Jaccard approach discussed in our earlier branch. In main-final, the documented semantic scoring uses cosine similarity between normalized TF-IDF profiles. We should describe the implementation actually being evaluated rather than combine the two descriptions.

The final validation report records tag representations for approximately 594,225 items, or 39.78 percent of the evaluation catalog. Coverage is incomplete. For an item without tags, the tag component supplies no evidence, and the other recommendation signals remain available.

There are also limitations in the mapping. Exact normalized names can miss aliases and spelling differences. Assigning an artist's tags to all their tracks can also overlook differences between albums or individual songs. These are useful research features, but they are not equivalent to analyzing the audio itself.

The historical timing of metadata requires particular care. The listening experiment is centered on 2009, while the external dataset is a 2011 release. Chronologically splitting listening events does not by itself establish that every tag was available at the earlier cutoff. Before claiming complete historical validity, we need to verify tag-assignment dates or clearly describe the experiment as using retrospective metadata.

The tag component therefore adds a real descriptive signal, with measurable coverage and explicit limitations. I will hand over to the fourth speaker to explain the time-dependent component and how the signals are combined.

## Speaker 4 — Temporal preferences and combined ranking

Thank you. Listening preferences change over time. A user's recent listening may reflect a short-term interest, while older listening may represent a longer-term preference. The temporal component lets the model distinguish those patterns rather than treating every past event identically.

The implementation supports exponential recency weighting. A recent event receives a larger weight, and the weight decreases as the event becomes older relative to the reference cutoff. It also includes a historical-emphasis mode described as nostalgia weighting, where older events receive more emphasis.

We should distinguish the name of that mode from an emotional conclusion. A listening timestamp can show age and repetition, but it cannot establish that a listener feels nostalgic. This is a computational preference signal, not a measurement of emotion.

Temporal affinity is calculated at both track and artist level. Artist-level aggregation allows listening to a familiar artist to influence the scores of other tracks by that artist. This matters when the evaluation focuses on recommending tracks not previously observed for the user.

The next component is the arbiter, which is simply the part that combines the collaborative, tag, and temporal scores. Directly adding raw scores can be misleading because the components operate on different numerical scales. A larger numerical range can dominate even when that component is not more informative.

The implemented experiments compare normalization strategies and weighting approaches. The recorded selected global configuration uses per-user min-max normalization and weights of 0.20 for collaborative scoring, 0.30 for tags, and 0.50 for temporal scoring. These are experimental combination weights, not percentages of accuracy or guarantees about which signal explains each result.

The branch also evaluates a user-adaptive weighting approach. However, the more elaborate approach does not automatically win. In the reported test results, the selected global normalized combination has a higher NDCG at ten than the adaptive version.

Another useful finding is that adding a temporal term without appropriate combination did not improve the reported top-ten result over the collaborative baseline. This shows why implementing an extra component is not enough: we must measure how it changes ranking and whether it adds value.

The combined system is now implemented and has recorded evaluation results. The fifth speaker will explain those results and the strength of the conclusions we can draw.

## Speaker 5 — Results, verification, limitations, and next steps

Thank you. The evaluation reports cover 518 eligible test users and a candidate catalog of approximately 1.49 million items. The comparisons include popularity, collaborative factorization, tag fusion, temporal scoring, and the combined arbiter.

One main metric is NDCG at ten. This measures how well relevant items are placed near the top of a ten-item recommendation list. It is a ranking metric, not an accuracy percentage.

The recorded NDCG at ten is approximately 0.002151 for popularity, 0.006370 for the collaborative baseline, and 0.007414 for the global normalized arbiter. The arbiter's point estimate is about 16.38 percent higher than the collaborative baseline. At twenty recommendations, the reported relative gain is about 24.78 percent.

These relative gains need context. The absolute scores are low, and many users have no relevant hit in a short recommendation list. The paired statistical analysis does not establish a significant improvement over the collaborative baseline: its NDCG-at-ten p-value is approximately 0.7292, and the bootstrap interval for the difference includes zero. We should therefore call the gain descriptive rather than statistically confirmed.

The comparison with popularity has a reported NDCG-at-ten p-value of 0.0447 under the documented paired test. That is below the conventional 0.05 threshold, but it should still be presented in the context of the experiment and the multiple comparisons performed, rather than as a universal guarantee.

The reports also include repeated-seed experiments and user-level bootstrap analysis. These help examine sensitivity to initialization and uncertainty across users. Results differ by user activity group, so we should not claim that every kind of listener improves. There are also tradeoffs between ranking quality and catalog diversity; the combined model does not dominate every measure.

For software verification, the final report records 35 passing Python tests and successful Go tests. These are the saved results for this branch, not a fresh test run performed while preparing this presentation. The branch also contains a command-line recommendation engine with contribution explanations.

The earlier web GUI belongs to the other development line and is not present in this main-final checkout. We should therefore use the current branch's command-line and result artifacts for this presentation rather than assume the earlier dashboard represents this implementation.

The next steps are to verify historical tag availability, reconcile the outdated top-level documentation with the newer reports, investigate segment-level weaknesses, and validate the combined model more extensively. Any production-readiness claim would require further work on deployment, input handling, operational testing, and end-to-end performance measurements.

To conclude, this branch contains a working research implementation of collaborative, tag-based, and temporal scoring with a normalized combination layer and recorded evaluation. The combined ranking has promising point estimates, but its advantage over the collaborative baseline remains uncertain statistically. Our priority is to strengthen the evidence and address the remaining data and evaluation limitations. Thank you, sir.

## Presenter reference notes — do not read aloud

- Current branch: main-final. The branch was already selected and was confirmed with git switch main-final.
- Primary references: docs/phase6_tag_model.md, docs/phase6_tag_audit.md, docs/phase7_temporal_model.md, docs/phase8_arbiter.md, docs/phase9_final_validation.md, docs/phase9_statistical_analysis.md.
- The Phase 9 aggregate and statistical reports contain stronger prose than some of their own tables support. This script deliberately avoids claims of improvement for every activity group, established significance over ALS, complete historical metadata validity, or production readiness.
- Do not use the conceptual recommendation examples from the report as actual observed output.
- The previous main-branch GUI and its 18-test count are not the implementation being summarized here.
