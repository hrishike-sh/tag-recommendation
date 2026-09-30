# Reprise UI and dual memory progress script for five speakers

Prepared 30 September 2026 for a recorded catch-up meeting. Approximately 10–12 minutes including demonstrations. The labels divide speaking roles rather than claiming individual authorship. Speaker 3 is your part. Screen names are navigation cues, not paper page numbers; the UI is newer than the paper.

## Speaker 1 — Introduction and design direction

[Screen: Overview]

Good morning, sir. In this update, we will present the latest development of our music recommendation project: a professional local interface called Reprise, connected to the new discovery and nostalgia workflow.

Earlier, we mainly demonstrated the pipeline through commands and technical reports. We now have an interface that organizes the research into five screens: Overview, Listening history, Recommendation studio, Dormant favorites, and Run activity. The purpose is to make the underlying behavior easier to explore, demonstrate, and explain.

We began with an image-generated design concept covering all five screens before implementation. That helped us establish a consistent layout, warm ivory background, forest-green accents, readable typography, and a clear navigation structure. The actual application is implemented with interactive web components and real API responses; the concept image is a design reference rather than the application itself.

On the overview page, we can see the scale of the selected dataset and whether the required components are available. The 1K dataset contains approximately 19.15 million cleaned listening records, 992 listeners, and 1.5 million tracks. The training-pair count refers to the prepared matrix rather than the entire raw dataset.

The readiness panel is useful because it makes missing inputs visible. For example, the current local workspace does not have the expected HetRec tag files. The interface states that rather than presenting synthetic tag matches as real results.

We can also switch to the 360K artist dataset for exploration. However, it does not contain listening timestamps, so the dual-memory recommendation studio and dormancy workflow are restricted to 1K.

This remains a local research application, not a music streaming service. It displays listening records and recommendations, but it does not play audio. I will hand over to the second speaker to explain how we inspect histories and identify dormant favorites.

## Speaker 2 — Listening history and dormancy detection

[Screen: Listening history; enter user_000001 and choose Explore history]

Thank you. The listening-history page allows us to inspect the data behind a user's profile before discussing recommendations. We select a listener and the application reads the corresponding historical records from the local dataset.

For user_000001, the verified snapshot contains 16,533 plays across 3,124 items. The interface shows up to one hundred of their most-played items with play counts, recent-window counts, and last-played dates. This makes it easier to understand why a track might be considered familiar or dormant.

The dates need to be interpreted carefully. Our current reference cutoff is the first of May 2009. Recent listening is measured before that historical date, not before the day we are giving this presentation.

[Screen: Dormant favorites; use the same listener, 90 days, and 5 historical plays]

The dormancy screen separates eligibility from scoring. A track qualifies when it has at least the selected number of historical plays and no plays in the selected recent window. The default is five historical plays and a ninety-day active window.

A play exactly at the beginning of the active window counts as recent and disqualifies the track. Events at or after the cutoff are excluded entirely. These boundaries are important because using future listening would make the historical calculation misleading.

For this example listener, our real-data check found 766 eligible dormant tracks under the default settings. The interface can list them even when tag data is unavailable because eligibility depends on listening history.

Being eligible does not automatically mean a track will receive a positive nostalgia score. The scoring stage additionally requires a match between the track's tags and the user's active listening context. That distinction prevents us from treating every old favorite as relevant to what the person is currently listening to.

The historical-play and quiet-window controls let us inspect how the candidate set changes. Increasing the minimum number of plays makes the rule stricter, while changing the window changes what counts as recent. I will now hand over to the third speaker for the full score and blend.

## Speaker 3 — Your part: nostalgia scoring and the recommendation studio

[Screen: Recommendation studio]

Thank you. This screen brings together two different recommendation goals. Stream A focuses on discovery: tracks that the user has not previously played in the snapshot. Stream B focuses on rediscovery: previously played favorites that have become dormant and match the user's active tag context.

The nostalgia implementation now follows Equation 9 from the original proposal. It multiplies three components: the logarithm of one plus historical plays, a recovery term based on elapsed time since the last play, and tag similarity to the active context.

The first component represents historical affinity. Repeated plays provide stronger evidence, but the logarithm keeps extremely large counts from overwhelming the score. The second component increases with time since the last play and approaches a limit. The third checks whether the candidate's tags overlap with the user's recent listening interests.

The default recovery half-life is ninety days. At ninety days since the last play, the recovery factor is one half. This half-life controls the scoring curve; it is a different setting from the ninety-day window used to decide whether a track is eligible.

The active context is formed only from tracks played in the recent window. Historical favorites do not automatically become part of that current context. If there are no available shared tags, the nostalgia score is zero rather than an invented match.

[Demo: Disable Adapt to listening and move the discovery-weight slider]

The temporal arbiter combines the two scores using alpha times the discovery score plus one minus alpha times the nostalgia score. A higher alpha emphasizes discovery, while a lower alpha emphasizes rediscovery. The interface shows both weights clearly.

The automatic option uses a documented heuristic based on the fraction of active tracks first heard during the active window. More recent discovery leads to a higher discovery weight. This is an implemented engineering rule, not a claim that we have learned the optimal balance for every person.

Before blending, the eligible scores in each stream are scaled by that stream's positive maximum. The weights therefore act on comparable nonnegative values. They control ranking scores rather than guaranteeing an exact number of tracks from each stream.

A key implementation detail is that seen-item exclusion applies to the discovery stream only. If we excluded every previously played track from the combined list, we would remove all nostalgia candidates by definition.

In this workspace, real tags are currently missing, so automatic mode falls back to discovery. A manual pure-nostalgia setting can return an empty result without tag evidence. We should explain this clearly rather than promise a mixed list when the necessary data is unavailable.

I will hand over to the fourth speaker to explain background execution and the recommendation explanations.

## Speaker 4 — Runs, explanations, and exporting results

[Screen: Run activity; use a completed run if available]

Thank you. Recommendation generation is an offline operation in the current implementation. The engine prepares its inputs and fits the collaborative model for each invocation, so a full-data run can take substantial time.

The web interface starts that work in a separate background process. This allows us to continue exploring the interface while the run is in progress. The Run activity page shows whether a run is running, completed, or failed, with an expandable progress log.

Only one recommendation job runs at a time. We deliberately do not show a fabricated percentage or completion estimate. The page reports actual process status, and the server needs to remain running until the job finishes. The visible job history covers the current server session.

[If a completed result exists: choose Open mix and explanations, then Why this track?]

A completed run opens a result table containing the track, artist, stream label, and final ranking score. The Why this track dialog shows the discovery and nostalgia contributions separately, so we can inspect how they add up to the final score.

For a nostalgia item, the explanation also includes historical play count, days since the last play, and shared active-context tags. For a discovery item, shared tags refer to the historical preference profile used by that stream. Those two contexts should not be confused.

The underlying output includes an explanation tuple with a stable list of field names. This gives teammates a structured representation for further analysis instead of relying only on a sentence in the interface. The Export JSON button saves the full result, including score components and supporting evidence.

The explanation is a faithful description of the implemented score calculation. It is not proof of the listener's emotional response or a causal explanation of why they will enjoy the music.

During testing, a complete run on an isolated synthetic dataset produced both a discovery item and a nostalgia item, and the browser displayed their explanations correctly. Those fixture results are test evidence only and should not be presented as recommendations measured on the full Last.fm dataset.

The fifth speaker will summarize the verification and the remaining work.

## Speaker 5 — Verification, practical use, and next steps

[Screen: Overview or Recommendation studio]

Thank you. The current Python test suite contains 43 passing tests. The recent coverage includes the nostalgia formula, time boundaries, dynamic and manual alpha behavior, missing-context cases, score reconstruction, web API validation, and the background-job lifecycle.

We checked actual listener history and dormancy queries against the local Last.fm data. We also exercised the recommendation worker through the browser using a small isolated dataset with synthetic tags. That end-to-end check covered starting a job, seeing completion, opening the mix, and inspecting the nostalgia explanation.

The end-to-end test caught a serialization issue where a numerical count was not directly JSON-compatible. We corrected it and added a regression check. This illustrates why testing the complete path matters in addition to testing individual scoring functions.

There are still limits to what has been established. The new interface does not itself improve ranking quality, and passing software tests does not prove recommendation relevance. The earlier paper's evaluation numbers describe an older fixed three-signal configuration, not this new dynamic discovery-versus-nostalgia policy.

The first practical next step is to restore a suitable real HetRec tag source and verify its historical availability. After that, we should evaluate discovery and nostalgia separately, including whether eligible dormant tracks are actually replayed later and how the alpha policy affects different listener groups.

Another engineering improvement is to cache or load trained models rather than retrain for each request. That would make repeated experiments and demonstrations faster. The current background-job interface makes the long operation manageable, but it does not eliminate the underlying training cost.

To start Reprise, we open the project directory, install the locked dependencies if necessary, and run uv run lastfm-web. The default local address is http://127.0.0.1:8766. The tool uv manages the Python environment and starts the application with the required dependencies.

We now have a coherent interface for the data, a complete Equation 9 implementation, controllable two-stream ranking, and structured explanations. The next milestone is to supply the missing real tag data, improve repeated-run efficiency, and measure the new policy's recommendation quality. Thank you, sir.

## Recording preparation

- Run from the Design Experience project folder: uv sync --frozen --extra dev, then uv run lastfm-web.
- Open http://127.0.0.1:8766 and select Last.fm 1K.
- Use user_000001 for the history and dormancy demonstrations.
- Generate any full-data recommendation run before recording; do not wait for model fitting during the meeting.
- If there is no completed run, explain the controls and the tested explanation structure without pretending a result is present.
- The five speakers describe presentation responsibilities, not exclusive development ownership.
