# Dual-memory recommendation contract

Implemented September 30, 2026 on main-final. Source: proposal section 2.2.4,
Equation 9 (page 4), and architecture alpha blend (page 3). This is distinct from
the paper's old three-signal recency arbiter; those research classes remain intact.

## Eligibility and Equation 9

For cutoff T and recent window W, history uses only events before T. Historical
plays precede T-W; active plays are in [T-W,T). A dormant item has at least tau
historical plays, zero active plays, and last_played_at < T-W. Defaults: W=90 days,
tau=5. A play exactly at the lower window boundary disqualifies the item. Events
at or after T have no effect on its counts, last play, context, or dynamic alpha.

Nostalgia = log1p(historical_plays) * (1-exp(-lambda*days_since_last_play)) * SimTag.
Lambda = ln(2)/half_life_days, default half-life 90 days. Elapsed dormancy is defined
here as time since last play, not time since crossing the eligibility boundary.

SimTag follows proposal Equation 7: Jaccard of present tag sets multiplied by the
mean candidate relevance on their shared tags. The context is the UNION of tags
from tracks played in the active window; historical favorites do not contribute
to this context. Existing nonnegative TF-IDF rows are L1-normalized for candidate
relevance. The upstream branch retains its smoothed IDF and artist-level features;
this does not replace those with acoustic features or the proposal's unsmoothed IDF.
No active context, missing tags, or disjoint tags means score zero. Eligibility is
retained for inspection but zero-evidence candidates are not recommended.

`listening_history` and `score_dormant` in `lastfm.nostalgia` expose the per-user
counts and all eligible candidates independently of fitting a model. 360K is not
supported for dormancy because it has no event timestamps.

## Two-stream alpha arbitration

Stream A retains this branch's normalized collaborative score plus historical
confidence-weighted cosine tag affinity. It only admits unseen items. Stream B
admits eligible, previously consumed dormant items. Seen-item masking is NOT
applied globally: that would remove every nostalgia candidate.

Before combination, each nonnegative stream is divided by its positive maximum
within its eligible set; an all-zero stream remains zero. The API includes these
scale factors and raw Equation 9 evidence so scores can be reconstructed.

Final = alpha * scaled_A + (1-alpha) * scaled_B.

The proposal does not specify an alpha update rule. The implemented explicit
heuristic is alpha = 0.2 + 0.6 * active_discovery_fraction, where that fraction is
the number of distinct active tracks first heard within the active window divided
by all distinct active tracks. No active history uses fraction 0.5. This heuristic
has not been tuned or evaluated for quality. Automatic mode sets alpha=1 when B
has no positive candidates, or alpha=0 when only B has positive candidates.
An explicit --alpha overrides this policy, including pure nostalgia at 0 and pure
discovery at 1; it can therefore return an empty list when its stream is unavailable.

Top-K selects positive-score candidates, deduplicates by catalog membership, and
uses item index for deterministic ties. K larger than the candidate count is safe;
K=0 returns an empty list. Dynamic alpha influences scores, not a guaranteed quota.

## Explainability and CLI

`recommend` defaults to `--mode dual`. `--mode global` retains the older experiment;
--weights applies only in global mode. The prepared CLI cutoff is May 1, 2009 UTC,
and it fits on the existing split's train-plus-validation snapshot before that date.
It still rebuilds matrices/fits ALS per call; no low-latency serving claim is made.

Every result contains item identity, rank, kind, reason, unrounded score, stream
contributions, and an explanation tuple with a response-level `explanation_fields`:

(kind, item_id, stream_a, stream_b, alpha, contribution_a, contribution_b,
historical_plays, active_plays, days_dormant, shared_tags)

JSON represents tuples as arrays. Contributions sum to the final score. Nostalgia
also includes raw score, log-play factor, recovery factor, tag similarity, shared
tag indices, and first/last play timestamps. Novelty shared tags refer to the
historical user profile; nostalgia shared tags refer only to active context.
These explain score construction, not emotional attachment or causation.
Unknown users fail explicitly instead of returning another user's recommendations.
Metadata is resolved from the ingestion report instead of a hard-coded snapshot.

Dual mode filters HetRec date fields strictly before the same cutoff BEFORE tag
frequency filtering and IDF calculation. Date-only assignments are interpreted at
UTC midnight. This prevents obvious future assignments from entering context;
it does not independently establish the historical validity of all artist aliases.
Legacy global mode deliberately preserves its original retrospective tag behavior.
Missing local HetRec assignments produce a zero-column tag matrix and output
`tags_available=false`; no network download or fabricated tags are attempted.

## Validation

Tests cover hand-calculated Eq. 9 and half-life, exact window boundaries, exclusion
of post-cutoff events, threshold behavior, active-only context and Eq. 7 relevance,
alpha endpoints and automatic adaptation, missing evidence, ties, empty/oversized
requests, JSON serialization and score reconstruction, dated tag filtering, and
engine wiring with a controlled model fixture. Full-data model quality has not
been re-evaluated; old paper metrics do not measure this new dual-memory policy.
