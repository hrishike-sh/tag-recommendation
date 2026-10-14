# Five person service and evaluation meeting script

This recorded catch-up covers the shared recommendation service, the new evaluation lab, and the first HNSW retrieval implementation. Allow approximately 12 to 15 minutes. Speaker 3 is your part, focusing on the evaluation metrics and how to interpret them.

## Before recording

Run `uv sync --extra dev --extra evaluation --extra ann`, then `uv run --no-sync lastfm-service prepare` and `uv run --no-sync lastfm-web` from the project directory. Open `http://127.0.0.1:8766/` and select Last.fm 1K. The first model fit can take several minutes, so prepare it before the meeting.

Generate one recommendation mix, one quality evaluation, and one latency benchmark before recording. Keep that server running so completed jobs remain visible in Run activity. Use the same listener for history, dormancy, and the mix. Check the Artist tags readiness indicator: if tags are absent, explain the fallback rather than promise nostalgia recommendations. A synthetic demonstration must be introduced as synthetic. Real reduced-model smoke results must be introduced as an eight-factor, one-iteration pipeline check.

Do not read the stage directions aloud. The spoken paragraphs can be read directly.

## Speaker 1 Project update and shared service

**Screen cue: Overview. Point to dataset statistics and Workspace readiness. Then open Recommendation studio.**

“Good morning, sir. In this update, we have moved the project from separate recommendation components toward one reusable service. Our system still has two main goals: helping a listener discover unfamiliar music and helping them return to meaningful tracks from their earlier history. The change is that these goals now run through the same service interface, with a shared model snapshot and one explanation format.

“Previously, a recommendation request could repeat the preparation and model-fitting work. That made it difficult to demonstrate the system efficiently and also made it harder to ensure that evaluation was testing the same code that the interface used. We now prepare a model bundle once. Later requests load that bundle and use it for recommendations. The browser, command-line entry point, and evaluation harness all use the same RecommendationService.

“The saved bundle contains the collaborative factors, sparse preference and confidence matrices, tag features, and the mappings between model indices and real item identifiers. These mappings matter because a score is only useful if it remains attached to the correct track. The bundle also identifies the training snapshot and cutoff. It lives locally under artifacts and is not uploaded with the source code.

“On the Overview screen, we can check whether the listening dataset, preference matrix, tags, and temporal history are available. We should interpret those checks separately. Having listening data does not automatically mean that tags are available. If the required tag evidence is missing, the system can still make discovery recommendations, but the nostalgia stream cannot manufacture semantic evidence.

“On Recommendation studio, both streams are presented as one mix. We can choose a listener, request a list size, and select an automatic or manual balance. The interface also now exposes an experimental HNSW retrieval option. Exact ranking remains the default, and the retrieval change will be explained by Speaker 4.

“The current service is a local research application. Each browser job uses a separate worker process and reloads the prepared bundle, so we have removed repeated fitting but have not claimed that startup and data loading cost nothing. The next speaker will explain how the two streams are combined and how we check the evidence behind a returned track.”

## Speaker 2 Stream integration and explainability

**Screen cue: Listening history for the selected listener. Move to Dormant favorites. Open the completed mix from Run activity, then click Why this track? in Recommendation studio.**

“First, this listening-history screen shows the evidence available before our historical cutoff. Stream A uses collaborative factors and available artist tags to rank tracks the listener has not heard. Stream B has a different purpose: it considers tracks that were heard enough times historically but have not been played during the recent window.

“The dormant-favorites screen makes that eligibility rule visible. With the default settings, a track needs at least five historical plays and no plays in the recent ninety-day window. We also check that its last play occurred before the window began. This prevents a recently heard song from being treated as a forgotten favorite merely because its total play count is high.

“For eligible tracks, Equation 9 combines three quantities. The first is the logarithm of one plus historical plays, which represents the strength of the old listening relationship without allowing extremely large counts to dominate. The second is a recovery term that increases with time since the last play. The third is tag similarity with the listener’s active context. A track with no supporting tag overlap receives no positive nostalgia score from this equation.

“Both streams then enter the dynamic temporal arbiter. The final score is alpha times the normalized discovery score plus one minus alpha times the normalized nostalgia score. Automatic alpha depends on how much of the recent listening consists of newly encountered tracks. If only one stream has positive evidence, automatic mode falls back to that available stream. A manually selected alpha remains the user’s choice.

“One integration detail is especially important. We must not apply a blanket seen-item filter after combining the streams. That would remove every nostalgia candidate because those tracks are intentionally familiar. The combined candidate set therefore includes unheard discovery items and eligible dormant items. Active familiar tracks remain outside this particular recommendation policy.

“When I click ‘Why this track?’, the explanation shows the score contributions and the available evidence. For a nostalgia recommendation, that includes historical plays, days since the last play, and shared tags. The contributions add up to the final score. These values explain the computation; they are not proof of an emotional reaction from the listener. We can export the result as JSON for teammates to inspect the same evidence. Speaker 3 will now show how we measure whether these recommendations recover useful future listening.”

## Speaker 3 Evaluation harness and metrics

**Screen cue: Evaluation lab. Explain the experiment controls. Open the completed Quality evaluation in Run activity, then remain on its metric table. Point to each column and the listener count underneath it.**

“My part is the evaluation harness. We chose scikit-learn for this implementation. It provides the DCG calculation used in NDCG and the cosine-distance calculation used in tag diversity. Our own harness supplies the recommendation-specific rules: the chronological split, the eligible candidates, novelty, nostalgia recall, and the way results are aggregated. We are not claiming that a RecBole experiment was run.

“The first thing to check is the timeline. The service is trained on listening events before the first of May 2009. The default test period starts on that date and ends before the first of July. We also filter historical tag assignments before the cutoff. This makes the question meaningful: using information available at that point, can the system recommend tracks that the listener plays later?

“We evaluate within the known training catalog. Future-only users and items are excluded and counted because the fitted model does not have representations for them. For each listener, relevant test targets are future plays of eligible unheard tracks or eligible dormant tracks. We do not treat an active, recently heard track as a missed recommendation when our policy intentionally excludes it. Users with no eligible test targets are reported separately.

“The first metric is NDCG at K, or normalized discounted cumulative gain. It rewards relevant recommendations and gives more credit when they appear near the beginning of the list. A correct track at rank one matters more than the same track near the bottom. We divide by the ideal ranking score so that the result is normalized. Unfilled list positions contribute no gain.

“The second metric is novelty. Here, novelty means popularity surprise, measured in bits. We calculate the proportion of training listeners who heard each recommended track, with smoothing, and take its negative base-two logarithm. Less common tracks contribute more novelty. This is different from the discovery label in the interface: a track can be new to a particular listener while still being globally popular. High novelty alone does not prove that the recommendation is relevant.

“The third metric is diversity. We calculate average pairwise cosine distance between the tag vectors of recommended tracks. If all tracks have very similar tags, diversity is low. If the tag representations differ, the score increases. We calculate this only where tag evidence exists and show tag coverage alongside it. With fewer than two tagged recommendations, diversity is unavailable; we display a dash instead of inventing a score.

“The fourth main metric is nostalgia recall. We first identify tracks that were dormant at the training cutoff and were actually replayed in the future test period. These are the nostalgia targets. We then ask what fraction appeared in the recommendation list. For example, retrieving two out of four such tracks gives a nostalgia recall of one half.

“The denominator does not depend on whether our model gave a target a positive score. This prevents the evaluation from hiding difficult targets when tags are missing. If targets exist but we retrieve none, recall is zero. If a listener has no nostalgia targets at all, the metric is unavailable for that listener. The counts under the columns show how many listeners contributed to each average.

“Finally, the quality run uses a reproducible listener sample, and the JSON export includes the cohort, exclusions, settings, and per-user values. We need to compare runs using the same protocol and cohort. These metrics give complementary evidence about ranking quality, popularity, variety, and rediscovery; none of them alone establishes that the full system is successful.”

## Speaker 4 HNSW retrieval and latency

**Screen cue: Recommendation studio, Retrieval dropdown. Then Evaluation lab, choose Exact vs HNSW latency. Open the completed Latency benchmark in Run activity and point to overlap, p50, and p95.**

“Our next development is the initial FAISS HNSW index. FAISS is a vector-search library, and HNSW is a graph-based approximate nearest-neighbor method. We use it to retrieve promising collaborative candidates without calculating every collaborative item score for every request.

“The index stores the model’s item-factor vectors. Our collaborative model scores an item using an inner product between the listener vector and the item vector, so the index uses that same inner-product objective. We do not normalize the vectors into cosine similarity because their magnitudes are part of the model’s scoring behavior.

“The retrieved collaborative shortlist is filtered to remove tracks already heard by the listener. We also include strong tag-based candidates, because a track can have useful semantic relevance even if the collaborative index did not retrieve it. After that, the shared service scores and blends the candidates. All eligible dormant candidates remain available to Stream B; the HNSW shortlist does not erase familiar tracks that the nostalgia stream needs.

“This is an approximate path. It can miss candidates, and its score normalization is based on the retrieved shortlist with a zero anchor instead of the full collaborative catalog. Therefore, we should not assume that the final list is identical to exact ranking. That is why the interface marks it experimental and why exact retrieval is still the default.

“The latency benchmark runs both methods for the same listeners. It first warms the history and the retrieval paths, then repeats the measurements and alternates their order. We report p50 latency, which represents the median, and p95 latency, which describes the slower end of the measured requests. We also report top-K overlap with the exact list, so a faster method cannot be judged only by speed while ignoring changes in what it recommends.

“The timing boundary is important. These are warm ranking measurements. They include retrieval, scoring, blending, and explanations, but exclude fitting, process startup, model loading, history input/output, and the first index build. Index preparation is reported separately. They are not browser-response or network-latency measurements.

“We have started the indexing infrastructure and made it measurable. We still need tuning and broader experiments before choosing an index configuration or claiming a consistent speed advantage. The rest of the pipeline, including tag processing and blending, can remain a significant cost. Speaker 5 will finish with the verification work and the next steps.”

## Speaker 5 Verification and handover

**Screen cue: Run activity and the completed job logs. Open the repository’s `docs/service-evaluation.md`, then the compact validation summary in `results/service/validation-summary.json`. End on Evaluation lab.**

“The final part of this update is how we checked the implementation and what teammates can reproduce. The automated tests cover the existing pipeline and the new shared-service path. The new tests check that loading a prepared model does not call training again, that items and tags from the future do not enter the training representation, and that stream contributions match the final recommendation score.

“We also use small examples with known answers to verify the metrics. This includes a hand-calculated NDCG result, novelty in bits, diversity for orthogonal tag vectors, and nostalgia recall for a replayed dormant track. Empty lists and missing tag evidence have explicit behavior. The retrieval tests check seen-item exclusion, duplicate avoidance, and loading a saved FAISS index.

“The interface has been exercised end to end using an isolated synthetic dataset. A quality job runs in the worker, completes in Run activity, and opens a metric table in Evaluation lab. The same flow supports the latency benchmark and recommendation explanations. Synthetic results show that the components work together; they must not be presented as performance on the Last.fm population.

“We have also used a reduced real-data model to check the pipeline against the full pre-cutoff catalog. That smoke model has eight factors and one training iteration. Its reports are separate from the default sixty-four-factor, ten-iteration model. The validation summary records the cohort sizes and measured results, so we can state exactly what was tested instead of mixing this run with the earlier research experiments.

“In the twenty-listener real-data smoke check, NDCG at ten was zero, so this run does not establish useful ranking quality. Tags were unavailable, making diversity undefined and nostalgia recall zero. The three-listener latency check had complete top-ten overlap with exact ranking, but HNSW had a slightly better median and a worse p95. That is not evidence of a consistent speed improvement.

“For the handover, the service and evaluation guide contains the commands, metric definitions, timeline, cache behavior, and indexing limitations. Source code and compact summaries are in the repository. Raw listening data, fitted models, vector indexes, and listener-level output remain local and ignored by Git. A teammate needs to prepare the local dataset and install the evaluation and ANN extras before reproducing a run.

“The main remaining research work is to run the default model over a larger fixed cohort, verify tag coverage, compare exact and approximate retrieval across candidate budgets, and assess quality together with latency. We should also distinguish evidence that a track was replayed from evidence that it caused nostalgia. The first can be evaluated from this dataset; the second would require additional user research.

“With this update, recommendation generation, explanations, quality evaluation, and retrieval benchmarking all have a shared service path and visible controls in the application. We can now demonstrate the workflow and make the next decisions using recorded measurements.”
