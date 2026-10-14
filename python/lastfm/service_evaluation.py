"""Chronological evaluation of the shared service using scikit-learn metrics.

Novelty uses training listener prevalence; diversity uses observed tag rows only.
Nostalgia recall targets dormant-at-cutoff tracks replayed during the holdout.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import duckdb
import numpy as np
from sklearn.metrics import dcg_score
from sklearn.metrics.pairwise import cosine_distances

from .ingest import literal
from .nostalgia import score_dormant, utc_cutoff


def ranking_metrics(recommended, positives, dormant_positives, prevalence, n_users, tags, k):
    if type(k)!=int or k<1: raise ValueError('k must be positive')
    ranked = list(recommended[:k])
    if len(set(ranked))!=len(ranked): raise ValueError('Recommendations must be unique')
    positives, dormant_positives = set(positives), set(dormant_positives)
    # Explicit rank order avoids tie averaging. Padding gives short lists zero gain.
    width = max(k,2)
    labels = np.zeros(width)
    labels[:len(ranked)] = [i in positives for i in ranked]
    ideal = np.zeros(width); ideal[:min(k,len(positives))]=1
    scores = np.arange(width,0,-1)[None,:]
    dcg = dcg_score(labels[None,:],scores,k=k,ignore_ties=True)
    idcg = dcg_score(ideal[None,:],scores,k=k,ignore_ties=True)
    novelty = float(np.mean([-np.log2((prevalence[i]+1)/(n_users+1)) for i in ranked])) if ranked else None
    known = [i for i in ranked if tags.getrow(i).nnz]
    diversity = None
    if len(known)>=2:
        distances = cosine_distances(tags[known])
        diversity = float(distances[np.triu_indices(len(known),1)].mean())
    return dict(ndcg=float(dcg/idcg) if idcg else None,
        novelty_bits=novelty, diversity=diversity,
        tag_coverage=len(known)/len(ranked) if ranked else 0.,
        nostalgia_recall=len(set(ranked)&dormant_positives)/len(dormant_positives) if dormant_positives else None,
        nostalgia_targets=len(dormant_positives), relevant_targets=len(positives), returned=len(ranked))


def holdout(service, test_end):
    end = utc_cutoff(test_end)
    if end<=utc_cutoff(service.cutoff): raise ValueError('test_end must be after training cutoff')
    events = literal((service.snapshot/'events/**/*.parquet').as_posix())
    with duckdb.connect() as db:
        rows = db.execute(f'''SELECT DISTINCT user_id,item_id FROM read_parquet({events},hive_partitioning=true)
            WHERE played_at>=? AND played_at<? ORDER BY user_id,item_id''',
            [utc_cutoff(service.cutoff),end]).fetchall()
    truth = {}; unknown_users=0; unknown_items=0
    for user,item in rows:
        if user not in service.user_index: unknown_users+=1; continue
        if item not in service.item_index: unknown_items+=1; continue
        truth.setdefault(user,set()).add(service.item_index[item])
    return truth,dict(holdout_pairs=len(rows),excluded_unknown_user_pairs=unknown_users,
                      excluded_unknown_item_pairs=unknown_items)


def evaluate_service(service, k_list=(5,10,20), test_end='2009-07-01T00:00:00Z',
                     max_users=20, retrieval='exact', candidates=500, **settings):
    if not k_list or any(type(k)!=int or k<1 for k in k_list): raise ValueError('K values must be positive integers')
    if max_users is not None and (type(max_users)!=int or max_users<1): raise ValueError('max_users must be positive or None')
    if retrieval not in ('exact','hnsw'): raise ValueError('Unknown retrieval mode')
    preparation_ms=0.
    if retrieval=='hnsw' and service._index is None:
        from .retrieval import CandidateIndex
        started=time.perf_counter()
        service._index=CandidateIndex(service.item_factors,service.folder/'hnsw.faiss')
        preparation_ms=(time.perf_counter()-started)*1000
    truth, exclusions = holdout(service,test_end)
    # Reproducible random cohort, independent of model outcomes.
    users = sorted(truth)
    if max_users is not None and len(users)>max_users:
        users = sorted(np.random.default_rng(42).choice(users,size=max_users,replace=False).tolist())
    prevalence = np.asarray((service.pref>0).sum(axis=0)).ravel()
    per_user=[]; latencies=[]; no_targets=0
    for user in users:
        history=service.history(user,settings.get('recent_days',90))
        dormant=score_dormant(history,service.items,service.tags,service.cutoff,
            settings.get('recent_days',90),settings.get('min_historical_plays',5),settings.get('half_life_days',90))
        seen={service.item_index[r['item_id']] for r in history if r['item_id'] in service.item_index}
        relevant=(truth[user]-seen)|(truth[user]&set(dormant))
        nostalgic=truth[user]&set(dormant)  # Do not require a positive model score.
        if not relevant: no_targets+=1; continue
        result=service.recommend(user,k=max(k_list),retrieval=retrieval,candidates=candidates,**settings)
        ranked=[service.item_index[r['item_id']] for r in result['recommendations']]
        latencies.append(result['latency_ms'])
        for k in k_list:
            per_user.append(dict(user_id=user,k=k,**ranking_metrics(ranked,relevant,nostalgic,
                                prevalence,len(service.users),service.tags,k)))
    summaries={}
    for k in k_list:
        rows=[r for r in per_user if r['k']==k]; values={}
        for name in ('ndcg','novelty_bits','diversity','nostalgia_recall','tag_coverage'):
            observed=[r[name] for r in rows if r[name] is not None]
            values[name]=float(np.mean(observed)) if observed else None
            values[name+'_users']=len(observed)
        summaries[str(k)]=values
    return dict(kind='evaluation',created=datetime.now(timezone.utc).isoformat(),
        service_key=service.key,cutoff=service.cutoff,test_end=test_end,retrieval=retrieval,
        settings=settings,index_preparation_ms=preparation_ms,candidates=candidates,cohort_seed=42,cohort_limit=max_users,
        available_holdout_users=len(truth),sampled_users=len(users),evaluated_users=len(latencies),
        excluded_no_eligible_targets=no_targets,exclusions=exclusions,metrics=summaries,
        latency_ms={'p50':float(np.percentile(latencies,50)) if latencies else None,
                    'p95':float(np.percentile(latencies,95)) if latencies else None},
        tags_available=service.tags_available,per_user=per_user,
        protocol='Known training catalog; unseen or dormant candidates; repeated dormant holdout positives retained. Missing metrics are null, not zero.')


def benchmark_service(service, users=None, k=10, candidates=500, repeats=3):
    if repeats<1: raise ValueError('repeats must be positive')
    users=users or service.users[:10]
    rows=[]; build_start=time.perf_counter()
    from .retrieval import CandidateIndex
    service._index=CandidateIndex(service.item_factors,service.folder/'hnsw.faiss')
    index_ms=(time.perf_counter()-build_start)*1000
    for user in users:
        # Warm history and index before measuring steady-state requests.
        service.recommend(user,k=k,retrieval='exact',candidates=candidates)
        service.recommend(user,k=k,retrieval='hnsw',candidates=candidates)
        for repeat in range(repeats):
            outputs={}
            for mode in (('exact','hnsw') if repeat%2==0 else ('hnsw','exact')):
                start=time.perf_counter()
                out=service.recommend(user,k=k,retrieval=mode,candidates=candidates)
                outputs[mode]=(out,(time.perf_counter()-start)*1000)
            exact={r['item_id'] for r in outputs['exact'][0]['recommendations']}
            approx={r['item_id'] for r in outputs['hnsw'][0]['recommendations']}
            rows.append(dict(user_id=user,exact_ms=outputs['exact'][1],hnsw_ms=outputs['hnsw'][1],
                overlap_at_k=len(exact&approx)/len(exact) if exact else None))
    overlaps=[r['overlap_at_k'] for r in rows if r['overlap_at_k'] is not None]
    return dict(kind='benchmark',service_key=service.key,k=k,candidates=candidates,repeats=repeats,
        users=len(users),index_build_or_load_ms=index_ms,
        overlap_at_k=float(np.mean(overlaps)) if overlaps else None,
        latency_ms={mode:{f'p{q}':float(np.percentile([r[mode+'_ms'] for r in rows],q)) for q in (50,95)} for mode in ('exact','hnsw')},
        note='Warm end-to-end ranking; excludes model load/fit and history I/O. HNSW uses shortlist normalization and may change rankings. No speedup assumed.',rows=rows)


def write_report(report,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
