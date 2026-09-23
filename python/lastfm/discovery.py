"""Retrieve unseen ALS candidates, optionally rerank with artist/tag fusion."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from .snapshots import read_db
from .tags import TagIndex, proposal_overlap, weighted_jaccard


def artist_profile_cosine(candidate_artists, favorite_artists, plays):
    # Cosine against a weighted mean of one-hot primary-artist vectors.
    profile = {}
    for artist, weight in zip(favorite_artists, np.log1p(plays)):
        if artist is not None:
            profile[artist] = profile.get(artist, 0.) + float(weight)
    norm = math.sqrt(sum(value*value for value in profile.values()))
    return [profile.get(artist,0.)/norm if norm and artist is not None else 0.
            for artist in candidate_artists]


def recommend(root: Path, user_id: str, dataset="1k", model_dir=None, tags_dir=None,
              k=10, candidates=200, favorites=50, mode=None, tag_metric="proposal",
              content_weight=1., tag_weight=1.):
    if k < 1 or candidates < k or favorites < 1:
        raise ValueError("Require candidates >= k >= 1 and favorites >= 1")
    if tag_metric not in {"proposal","weighted-jaccard"}:
        raise ValueError("Unknown tag metric")
    if any(not math.isfinite(w) or w<0 for w in (content_weight,tag_weight)) or content_weight+tag_weight == 0:
        raise ValueError("Fusion weights must be finite, nonnegative and not both zero")
    mode = mode or ("fusion" if tags_dir is not None else "als")
    if mode not in {"als","fusion"}:
        raise ValueError("mode must be als or fusion")
    if mode == 'fusion' and tags_dir is None:
        raise ValueError("Fusion requires a real tag artifact; use mode=als until tags are imported")
    if model_dir is None:
        latest=json.loads((root/f'artifacts/reports/model-{dataset}.json').read_text())
        model_dir=Path(latest['path'])
    model_dir=Path(model_dir).resolve()
    metadata=json.loads((model_dir/'model.json').read_text())
    if metadata['dataset'] != dataset:
        raise ValueError("Model dataset mismatch")
    counts=load_npz(model_dir/'counts.npz').tocsr()
    users=np.load(model_dir/'user_factors.npy',mmap_mode='r',allow_pickle=False)
    items=np.load(model_dir/'item_factors.npy',mmap_mode='r',allow_pickle=False)
    if counts.shape != (len(users),len(items)) or list(counts.shape) != metadata['shape'] or users.shape[1] != items.shape[1]:
        raise ValueError("Model factors and mappings have inconsistent dimensions")
    with read_db() as db:
        row=db.execute('SELECT user_index FROM read_parquet(?) WHERE user_id=?',
                       [str(model_dir/'users.parquet'),user_id]).fetchone()
        if row is None:
            raise ValueError("Unknown training user; cold-start fallback is not implemented")
        user_index=int(row[0])
        history=counts.getrow(user_index)
        with threadpool_limits(limits=1,user_api='blas'):
            scores=np.asarray(items @ users[user_index],dtype=np.float64)
        scores[history.indices]=-np.inf
        eligible=np.flatnonzero(np.isfinite(scores))
        # Stable tie-breaking by item_index; score one user, never a dense user-item matrix.
        order=np.lexsort((eligible,-scores[eligible]))
        selected=eligible[order[:candidates]]
        preferred_order=np.lexsort((history.indices,-history.data))[:favorites]
        preferred=history.indices[preferred_order]
        plays=history.data[preferred_order]
        ids=[int(i) for i in np.concatenate([selected,preferred])]
        names={int(i):(item,None,None,None) for i,item in db.execute(
            'SELECT item_index,item_id FROM read_parquet(?) WHERE item_index=ANY(?)',
            [str(model_dir/'items.parquet'),ids]).fetchall()}
        if (model_dir/'item_metadata.parquet').exists():
            for i,item,artist,artist_name,track_name in db.execute(
                'SELECT item_index,item_id,artist_id,artist_name,track_name FROM read_parquet(?) WHERE item_index=ANY(?)',
                [str(model_dir/'item_metadata.parquet'),ids]).fetchall():
                names[int(i)]=(item,artist,artist_name,track_name)
    content_scores=artist_profile_cosine([names[int(i)][1] for i in selected],
                                         [names[int(i)][1] for i in preferred],plays)
    tags=None
    profile={}
    if tags_dir is not None:
        tags=TagIndex(Path(tags_dir),metadata['matrix_fingerprint'])
        if tags.counts.shape[0] != len(items):
            raise ValueError("Tag row count does not match model items")
        profile=tags.profile(preferred,plays)
    metric=proposal_overlap if tag_metric=='proposal' else weighted_jaccard
    results=[]
    for item_index, content in zip(selected,content_scores):
        item_index=int(item_index)
        item,artist,artist_name,title=names[item_index]
        item_tags=tags.item(item_index) if tags else {}
        tag_score=metric(item_tags,profile) if tags else 0.
        fusion=content_weight*content+tag_weight*tag_score
        common=sorted(item_tags.keys() & profile.keys(),key=lambda t:(-item_tags[t],t))
        shared=[tags.vocabulary[t] for t in common[:5]] if tags else []
        score=fusion if mode=='fusion' else float(scores[item_index])
        results.append({"item_index":item_index,"item_id":item,"artist":artist_name,"track":title,
                        "score":score,"als_score":float(scores[item_index]),
                        "artist_content_cosine":content,"tag_score":tag_score if tags else None,
                        "shared_tags":shared,"candidate_has_tags":bool(item_tags),
                        "reason":("Unseen track/artist selected by ALS; reranked using primary-artist cosine and tag relevance."
                                   if mode=='fusion' else "Unseen track/artist ranked by confidence-weighted collaborative affinity.")})
    results.sort(key=lambda r:(-r['score'],-r['als_score'],r['item_index']))
    return {"stream":"A","user_id":user_id,"dataset":dataset,"mode":mode,
            "model_path":str(model_dir),"matrix_fingerprint":metadata['matrix_fingerprint'],
            "cutoff":metadata['cutoff'],"seen_items_excluded":history.nnz,
            "candidate_pool_size":len(selected),"favorite_profile_items":len(preferred),
            "tag_metric":tag_metric if tags else None,"tag_profile_available":bool(profile),
            "content_features":metadata.get('content_features'),
            "fusion_weights":{"content":content_weight,"tags":tag_weight} if mode=='fusion' else None,
            "recommendations":results[:k]}
