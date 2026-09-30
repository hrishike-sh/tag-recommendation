# Five-person paper presentation — page-by-page script

Source: C:/Users/rushi/Downloads/main.pdf
Navigation: PDF viewer page positions 1–12, counting the title/abstract page as page 1
Reading guide: Page prefixes are scrolling cues; they do not need to be spoken aloud
Estimated duration: 12–15 minutes with pauses to point at figures
Speaker allocation: 1 = pages 1–3; 2 = page 4; 3 (you) = pages 5–6; 4 = pages 7–8; 5 = pages 9–12

## Speaker 1 — Motivation, background, and data | Pages 1–3

Page 1 — Good morning, sir, today we will walk through our paper on multi-stream music recommendation at extreme catalog scale.

Page 1 — The problem is to recommend a small number of useful tracks when the available catalog contains almost one and a half million candidates and each listener has interacted with only a small fraction of them.

Page 1 — Our system combines three signals: collaborative listening patterns, semantic information from artist tags, and temporal information about recent listening.

Page 1 — The main idea is that these signals produce scores on different scales, so we align the collaborative scores before combining the streams.

Page 1 — Our contribution is the evaluation of this sparse, normalized combination under full-catalog ranking, rather than a claim that we invented a new neural architecture.

Page 1 — The abstract reports a higher aggregate ranking score than the collaborative baseline, but also states that this difference is not statistically significant in the paired user-level comparison.

Page 2 — The related-work section explains why implicit feedback is different from explicit ratings, because not playing a track does not necessarily mean disliking it.

Page 2 — Collaborative models learn shared patterns across listeners, tags provide descriptive information, and temporal models account for changes in listening over time.

Page 2 — The score-fusion discussion identifies our central challenge: a stream with larger numerical values can dominate the final ranking even when its information is not more useful.

Page 2 — Another important choice is full-catalog evaluation, where relevant tracks compete against the entire eligible catalog instead of a small sampled set of alternatives.

Page 3 — Table I summarizes the data, including 19,150,865 clean listening events, 992 users, and 1,493,688 tracks in the evaluated candidate catalog.

Page 3 — The table also records 2,299 retained tags and 518 evaluable test users, which is the user count relevant to the reported test metrics.

Page 3 — Figure 1 shows the chronological protocol, with training before April 2009, validation during April, and testing during May and June.

Page 3 — The purpose of this ordering is to select settings using earlier observations and reserve later listening for the final evaluation.

Page 3 — At test time, tracks already present in the user's training or validation history are excluded so that the recommendations represent previously unconsumed catalog items.

Page 3 — The tag-alignment section explains that only about 39.78 percent of the evaluated catalog has tag representations, so missing semantic evidence is explicitly retained rather than invented.

Page 3 — I will now hand over to the second speaker to explain how the collaborative and tag streams turn these inputs into recommendation scores.

## Speaker 2 — Architecture, collaborative model, and semantic tags | Page 4

Page 4 — Figure 2 is the overall architecture diagram, and it shows the collaborative, semantic, and temporal streams remaining separate until the final combination stage.

Page 4 — This separation lets us inspect each component and later test what happens when a stream is removed.

Page 4 — In the collaborative stream, equation 4 defines preference as whether the user has played the track, while equation 5 assigns confidence based on the number of plays.

Page 4 — The confidence rule is one plus forty times the natural logarithm of one plus the play count.

Page 4 — The logarithm allows repeated listening to increase confidence without letting extremely large play counts grow in influence at the same linear rate.

Page 4 — The model learns a vector for each user and track using an implicit ALS-style factorization objective, and their dot product gives the collaborative score.

Page 4 — The configuration in this paper uses 64 latent factors, regularization of 0.05, and ten ALS iterations.

Page 4 — These factors are learned numerical features, so we should not describe each dimension as a particular genre or emotion.

Page 4 — On the right side of the page, Stream A introduces semantic information using tags associated with each track's artist.

Page 4 — Term frequency measures a tag's share of the assignments, while inverse document frequency reduces the influence of tags that are common across the catalog.

Page 4 — The resulting TF-IDF vector is normalized, allowing comparisons to depend on the tag mixture instead of simply the total number of assignments.

Page 4 — A user's semantic profile combines the tag representations of their historical tracks using confidence weights.

Page 4 — The semantic score is the dot product of normalized user and item tag vectors, which acts as cosine similarity.

Page 4 — This paper therefore uses cosine similarity for semantic scoring, rather than the weighted Jaccard formulation discussed in an earlier version of the project.

Page 4 — Artist tags provide useful descriptive evidence, but tracks by the same artist inherit the same underlying tag information and may still differ musically.

Page 4 — No waveform or acoustic representation is used here, so this component should be described as artist-tag similarity rather than audio similarity.

Page 4 — I will hand over to the third speaker to explain the temporal stream and how the three signals are combined.

## Speaker 3 — Your part: temporal scoring, normalization, and setup | Pages 5–6

Page 5 — The temporal stream models the idea that recent listening can be more relevant to a user's current interests than older listening.

Page 5 — Equation 13 calculates the age of an interaction in days relative to the evaluation reference time, rather than relative to today's date.

Page 5 — Equation 14 then multiplies the interaction confidence by an exponential decay term, so its contribution becomes smaller as it gets older.

Page 5 — The decay rate used in this paper is 0.10 per day, corresponding to a half-life of approximately 6.93 days.

Page 5 — In practical terms, the age-based weight roughly halves every seven days, although an older interaction can still retain a non-zero contribution.

Page 5 — The temporal vector is normalized, and the representation is stored sparsely to avoid allocating values for every possible user–track combination.

Page 5 — This paper evaluates temporal recency, so I would not describe this result as proof that the system recognizes emotional nostalgia.

Page 5 — The next section explains why normalization is needed before combining the streams.

Page 5 — Collaborative scores come from latent-vector dot products, while the semantic and temporal signals have already been normalized, so the raw numbers are not directly comparable.

Page 5 — Per-user min-max normalization rescales the collaborative score range independently for each listener, making the fusion weights less dependent on that listener's raw score spread.

Page 5 — The arbiter then calculates a weighted sum using 0.20 for collaborative scoring, 0.30 for semantic tags, and 0.50 for temporal scoring.

Page 5 — These values are combination weights selected using validation data, not accuracy percentages or a statement that temporal information is always the most important component.

Page 5 — After combination, previously consumed tracks are masked and exact top-K selection identifies the highest-scoring eligible items.

Page 5 — Exact retrieval means the selection follows the computed scores across the full eligible catalog, rather than depending on an approximate nearest-neighbor index.

Page 5 — The experimental-setup section introduces recall, precision, NDCG, and MAP at list lengths of five, ten, and twenty.

Page 5 — Recall measures how much held-out relevant music is recovered, precision measures how much of the recommendation list is relevant, and NDCG rewards placing relevant tracks nearer the top.

Page 6 — Figure 3 brings these steps together visually, from the separate signal scores through normalization and weighted combination.

Page 6 — Table II records the fixed settings, including the 64 factors, ten training iterations, 2,299 tags, decay rate, and final fusion weights.

Page 6 — Table III shows the validation weight grid, where the selected 0.20, 0.30, and 0.50 combination has the highest validation NDCG at ten among the listed choices.

Page 6 — Choosing the configuration on validation data and freezing it before testing avoids selecting weights simply because they look good on the final test results.

Page 6 — The reproducibility section distinguishes repeating the same seeded run from changing the initialization seed to measure sensitivity.

Page 6 — The paper reports three identical seed-42 metric runs and also evaluates five different seeds, because repeatability and robustness are different questions.

Page 6 — I will now hand over to the fourth speaker to explain the test results and how strongly the evidence supports the observed gains.

## Speaker 4 — Ranking results, user groups, and uncertainty | Pages 7–8

Page 7 — Table IV compares seven configurations on the same 518 test users and the same catalog, making the results directly comparable within this experiment.

Page 7 — The global normalized arbiter reaches NDCG at ten of 0.007414, compared with 0.006370 for the collaborative baseline and 0.002151 for popularity.

Page 7 — Figure 5 visualizes the aggregate NDCG comparison, with the global arbiter achieving the highest values in the locked seed-42 configuration.

Page 7 — The relative NDCG-at-ten gain over the collaborative baseline is approximately 16.38 percent, but this is a relative change in a ranking metric rather than a 16-percentage-point increase in accuracy.

Page 7 — The paper also reports a temporal-only fusion score of 0.006952, while tag fusion and fixed unnormalized fusion score below the collaborative baseline at NDCG at ten.

Page 7 — This shows that adding more signals does not automatically improve ranking, because their scaling and combination matter.

Page 7 — Table V and Figure 6 split the results by user activity, and the clearest descriptive benefit appears among low-activity listeners.

Page 7 — For that group, NDCG at ten increases from about 0.011578 to 0.017421, while the middle- and high-activity groups have lower scores under the global arbiter than under the collaborative baseline.

Page 8 — The continuation explains that one global set of weights is a compromise across users with different amounts of listening history.

Page 8 — The statistical evaluation compares paired results for the same users rather than treating each model's aggregate score as independent evidence.

Page 8 — Against the collaborative baseline, the Wilcoxon p-value is 0.7292, so the observed average gain is not statistically significant at the conventional 0.05 threshold.

Page 8 — Against popularity, the paper reports a p-value of 0.0447 for NDCG at ten, which meets that threshold under the reported test.

Page 8 — Figure 7 shows the bootstrap confidence interval for the paired difference, and its crossing of zero reinforces the uncertainty in the comparison with the collaborative baseline.

Page 8 — The bootstrap uses 10,000 resamples, but resampling does not remove the limitations of the underlying 518-user test population.

Page 8 — The seed analysis also shows that the arbiter varies more across initializations than the collaborative baseline on the reported NDCG-at-ten measure.

Page 8 — Our conclusion is therefore that the combined model has promising aggregate point estimates, while its superiority over the collaborative baseline is not statistically established.

Page 8 — I will hand over to the fifth speaker for component contributions, runtime, explainability, and the final limitations.

## Speaker 5 — Tradeoffs, runtime, explanations, and conclusion | Pages 9–12

Page 9 — Table VI collects the statistical comparisons, including the important distinction between the significant popularity comparison and the non-significant collaborative-baseline comparison.

Page 9 — Table VII shows that the arbiter recommends 6,118 unique tracks in the top-twenty evaluation, compared with 7,165 for the collaborative baseline and only 90 for popularity.

Page 9 — These results show a tradeoff, because the arbiter's higher aggregate ranking score does not also give it the highest catalog coverage.

Page 9 — The stream-removal experiment reduces NDCG at ten by about 80.90 percent when collaborative scoring is removed, showing that the collaborative component remains central in this configuration.

Page 9 — Removing the tag stream produces a smaller reported reduction of about 14.08 percent, indicating complementary semantic information within this experiment.

Page 9 — Table VIII separates offline preparation from query execution, with roughly 68 minutes for ALS training and 96.48 milliseconds for a single-user top-ten query under the recorded benchmark conditions.

Page 9 — Figure 8 explains the computational approach, using sparse structures and exact partial sorting without allocating a dense matrix for all users and all tracks.

Page 9 — These benchmark numbers describe the measured setup and should not be presented as a guaranteed response time on every computer or for the entire application startup process.

Page 10 — The explainability section breaks a final recommendation score into collaborative, semantic, and temporal contributions.

Page 10 — This tells us which terms contributed to the model's score, but it does not establish the psychological reason a listener would enjoy the track.

Page 10 — The discussion reiterates that normalization helps align different score ranges, while also noting that min-max scaling can be sensitive to extreme values.

Page 10 — It also emphasizes that the benefit is uneven across user groups and that full-catalog results should not be compared casually with studies using small sampled candidate sets.

Page 11 — The limitations include sparse test hits, incomplete tag coverage, artist-level rather than track-level semantics, and the absence of acoustic features.

Page 11 — The evaluation is offline and observational, so higher ranking metrics do not directly prove increased user satisfaction or causal benefits.

Page 11 — The global weights may also behave differently in other datasets or time periods, and testing five seeds does not establish generalization across all modeling choices.

Page 11 — Future work includes track-level audio features, better user-adaptive weighting, session-aware modeling, alternative calibration methods, and evaluations across more datasets or user studies.

Page 11 — The conclusion is that sparse multi-stream scoring and exact retrieval are feasible at this catalog size, with qualified evidence of ranking benefits under the tested conditions.

Page 11 — We should not turn that conclusion into a claim of universal superiority or a complete solution to cold-start recommendation.

Page 12 — Appendix A gathers the locked configuration so that the model settings can be reproduced without searching through the whole paper.

Page 12 — Appendix B lists the reproducibility controls, including chronological boundaries, fixed settings, repeated runs, multiple seeds, paired tests, bootstrap analysis, and runtime records.

Page 12 — Appendix C makes the final distinction between a descriptive metric difference, a statistically significant finding, and a practical observation about system behavior.

Page 12 — Our presentation follows that distinction by reporting both the positive aggregate results and the uncertainty surrounding them.

Page 12 — Thank you, sir, that completes our walkthrough of the paper.
