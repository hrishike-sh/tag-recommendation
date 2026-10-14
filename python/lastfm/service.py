"""Reusable dual-stream model snapshot, shared by CLI, web and evaluation."""
from datetime import timedelta
import csv
import hashlib
import json
from pathlib import Path
import time
import uuid

import duckdb
import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD
from .arbiter import normalize_scores
from .ingest import literal
from .nostalgia import DynamicTemporalArbiter, score_dormant, utc_cutoff, validate_parameters
from .split import split_train_val_test
from .tag_msvd import build_tag_matrix

CUTOFF = '2009-05-01T00:00:00Z'
SCHEMA = 1


class RecommendationService:
    """Fit once; serve both streams from the same immutable training catalog."""

    @classmethod
    def open(cls, root, *, factors=64, iterations=10, rebuild=False):
        root = Path(root).resolve()
        report = root/'artifacts/reports/ingest-1k.json'
        ingest = json.loads(report.read_text())
        snapshot = Path(ingest['snapshot'])
        if not snapshot.is_absolute(): snapshot = root/snapshot
        # Snapshots are immutable. Tag file stamps invalidate cached features.
        tagdir = root/'data/raw/hetrec2011-lastfm-2k'
        stamps = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                  for p in sorted(tagdir.glob('*.dat'))]
        spec = dict(schema=SCHEMA, ingest=ingest, tags=stamps, factors=factors,
                    iterations=iterations, cutoff=CUTOFF, seed=42, regularization=.05)
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
        folder = root/'artifacts/service'/key
        if (folder/'manifest.json').exists() and not rebuild:
            return cls.load(folder)
        started = time.perf_counter()
        print('Preparing shared training snapshot and fitting MSVD.', flush=True)
        split = split_train_val_test(root, dataset='1k',
            train_cutoff='2009-04-01T00:00:00Z', val_cutoff=CUTOFF,
            test_end='2009-07-01T00:00:00Z', kappa=40.)['test_split']
        service = cls()
        service.root, service.snapshot = root, snapshot
        service.cutoff, service.key = CUTOFF, key
        service.users, service.items = split['user_ids'], split['item_ids']
        service.pref, service.delta = split['P_train'], split['D_train']
        if not service.users or not service.items: raise ValueError('Training catalog is empty')
        model = ImplicitMSVD(factors=factors, iterations=iterations, regularization=.05, seed=42)
        model.fit(service.pref, service.delta, compute_loss_history=False, verbose=False)
        service.user_factors, service.item_factors = model.user_factors, model.item_factors
        meta = snapshot/'track_metadata.parquet'
        service.tags_available = (tagdir/'user_taggedartists.dat').exists()
        if service.tags_available:
            service.tags, vocabulary = build_tag_matrix(root, service.items, meta,
                                                        min_tag_freq=5, cutoff_iso=CUTOFF)
        else:
            service.tags = sp.csr_matrix((len(service.items), 0), dtype=np.float32)
            vocabulary = {}
        service.names = {}
        if vocabulary and (tagdir/'tags.dat').exists():
            with (tagdir/'tags.dat').open(encoding='latin-1') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    tag = int(row['tagID'])
                    if tag in vocabulary: service.names[vocabulary[tag]] = row['tagValue']
        with duckdb.connect() as db:
            rows = db.execute(f'SELECT item_id, artist_name, track_name FROM read_parquet({literal(meta.as_posix())}) ORDER BY 1,2,3').fetchall()
        service.metadata = {r[0]: r[1:] for r in reversed(rows)}
        service.fit_seconds = time.perf_counter()-started
        service._initialize()
        # Publish complete bundles atomically; simultaneous processes may fit redundantly.
        staging = folder.parent/(key+'-'+uuid.uuid4().hex)
        service.save(staging)
        if not folder.exists():
            try: staging.rename(folder)
            except FileExistsError: pass
        elif rebuild:
            # Rebuilds retain their own immutable bundle instead of overwriting readers.
            folder = staging
        service.folder = folder
        return service

    def _initialize(self):
        self.user_index = {u:i for i,u in enumerate(self.users)}
        self.item_index = {v:i for i,v in enumerate(self.items)}
        self._histories = {}
        self._index = None

    def save(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder/'factors.npz', users=self.user_factors, items=self.item_factors)
        for name in ('pref','delta','tags'): sp.save_npz(folder/f'{name}.npz', getattr(self,name))
        manifest = dict(schema=SCHEMA, root=str(self.root), snapshot=str(self.snapshot),
            cutoff=self.cutoff, key=self.key, users=self.users, items=self.items,
            names=self.names, metadata=self.metadata, tags_available=self.tags_available,
            fit_seconds=self.fit_seconds)
        (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.pop('schema') != SCHEMA: raise ValueError('Unsupported service bundle schema')
        service = cls()
        for name,value in manifest.items(): setattr(service,name,value)
        service.root, service.snapshot = Path(service.root), Path(service.snapshot)
        service.names = {int(k):v for k,v in service.names.items()}
        with np.load(folder/'factors.npz', allow_pickle=False) as arrays:
            service.user_factors, service.item_factors = arrays['users'], arrays['items']
        for name in ('pref','delta','tags'): setattr(service,name,sp.load_npz(folder/f'{name}.npz'))
        service.folder = folder
        service._initialize()
        return service

    def history(self, user, recent_days=90):
        key = (user,recent_days)
        if key not in self._histories:
            end = utc_cutoff(self.cutoff); boundary = end-timedelta(days=recent_days)
            events = literal((self.snapshot/'events/**/*.parquet').as_posix())
            with duckdb.connect() as db:
                rows = db.execute(f'''SELECT item_id,
                    count(*) FILTER (WHERE played_at < ?), count(*) FILTER (WHERE played_at >= ?),
                    min(played_at),max(played_at) FROM read_parquet({events},hive_partitioning=true)
                    WHERE user_id=? AND played_at < ? GROUP BY item_id ORDER BY item_id''',
                    [boundary,boundary,user,end]).fetchall()
            self._histories[key] = [dict(zip(('item_id','historical_plays','active_plays',
                'first_played_at','last_played_at'),r)) for r in rows]
        return self._histories[key]

    def recommend(self, user_query, k=10, alpha=None, recent_days=90,
                  min_historical_plays=5, half_life_days=90, retrieval='exact', candidates=500):
        validate_parameters(recent_days,min_historical_plays,half_life_days)
        if type(k)!=int or k<0: raise ValueError('k must be a nonnegative integer')
        if retrieval not in ('exact','hnsw'): raise ValueError('retrieval must be exact or hnsw')
        if type(candidates)!=int or candidates<max(k,1): raise ValueError('candidates must be at least max(k,1)')
        started = time.perf_counter()
        user = str(user_query)
        if user not in self.user_index and user.isdigit() and int(user)<len(self.users): user=self.users[int(user)]
        if user not in self.user_index: raise ValueError(f'Unknown user: {user_query}')
        u = self.user_index[user]
        profile = np.asarray(((self.pref.getrow(u)+self.delta.getrow(u))@self.tags).toarray()).ravel()
        norm = np.linalg.norm(profile)
        if norm: profile /= norm
        semantic = np.asarray(self.tags@profile).ravel()
        history = self.history(user,recent_days)
        if retrieval=='exact':
            a = normalize_scores(self.user_factors[u]@self.item_factors.T)+semantic
            candidate_count = len(self.items)
        else:
            from .retrieval import CandidateIndex
            if self._index is None:
                self._index = CandidateIndex(self.item_factors, self.folder/'hnsw.faiss')
            seen = {self.item_index[r['item_id']] for r in history if r['item_id'] in self.item_index}
            collab = self._index.search(self.user_factors[u], candidates, seen)
            # Union semantic candidates: collaborative retrieval alone misses strong tags.
            tag_candidates = np.flatnonzero(semantic>0)
            tag_order = tag_candidates[np.lexsort((tag_candidates,-semantic[tag_candidates]))]
            tagged = [int(i) for i in tag_order if i not in seen][:candidates]
            selected = np.array(sorted(set(collab)|set(tagged)),dtype=int)
            a = np.zeros(len(self.items))
            if len(selected):
                raw = self.item_factors[selected]@self.user_factors[u]
                a[selected] = normalize_scores(np.append(raw,0))[:-1]+semantic[selected]
            candidate_count = len(selected)
        dormant = score_dormant(history,self.items,self.tags,self.cutoff,recent_days,
                                 min_historical_plays,half_life_days)
        result = DynamicTemporalArbiter().recommend(a,self.items,history,dormant,self.cutoff,
                    recent_days,alpha,k,self.names)
        for rank,row in enumerate(result['recommendations'],1):
            idx = self.item_index[row['item_id']]
            if row['kind']=='novelty':
                shared = [int(t) for t in self.tags.getrow(idx).indices if profile[t]>0]
                shared.sort(key=lambda t:(-profile[t],t))
                names = tuple(self.names.get(t,f'tag:{t}') for t in shared[:10])
                row['explanation_tuple'] = row['explanation_tuple']._replace(shared_tags=names)
                if names: row['reason']='Unheard track with collaborative and artist-tag affinity; shared tags: '+', '.join(names)+'.'
            artist,track = self.metadata.get(row['item_id'],(None,None))
            row.update(rank=rank,artist_name=artist,track_name=track)
        result.update(user_id=user,user_index=u,model='Dual-memory temporal arbiter',
            cutoff=self.cutoff,recent_days=recent_days,min_historical_plays=min_historical_plays,
            half_life_days=half_life_days,k=k,dormant_candidates=len(dormant),
            positive_nostalgia_candidates=int(sum(r['score']>0 for r in dormant.values())),
            tags_available=self.tags_available,retained_tag_count=self.tags.shape[1],
            service_key=self.key,retrieval=retrieval,candidate_count=candidate_count,
            scoring_policy='catalog normalization' if retrieval=='exact' else 'approximate shortlist normalization with zero anchor',
            latency_ms=(time.perf_counter()-started)*1000)
        return result
