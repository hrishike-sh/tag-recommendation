"""FAISS HNSW maximum inner-product candidate retrieval (no vector normalization)."""
from pathlib import Path
import numpy as np

class CandidateIndex:
    def __init__(self, vectors, path=None, m=32, ef_construction=100, ef_search=128):
        try: import faiss
        except ImportError as exc: raise RuntimeError('Install ANN support: uv sync --extra ann') from exc
        self.vectors = np.ascontiguousarray(vectors,dtype=np.float32)
        path = Path(path) if path else None
        if path and path.exists():
            self.index = faiss.read_index(str(path))
            if self.index.ntotal!=len(vectors) or self.index.d!=vectors.shape[1]:
                raise ValueError('Index does not match the model bundle')
        else:
            self.index = faiss.IndexHNSWFlat(vectors.shape[1],m,faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.add(self.vectors)
            if path:
                import uuid
                temporary = path.with_suffix('.'+uuid.uuid4().hex+'.partial')
                faiss.write_index(self.index,str(temporary)); temporary.replace(path)
        self.index.hnsw.efSearch = ef_search

    def search(self, vector, k, excluded=()):
        if k<1: raise ValueError('k must be positive')
        excluded = set(excluded)
        count = min(len(self.vectors),max(k*2,k+len(excluded)))
        _,indices = self.index.search(np.ascontiguousarray(vector,dtype=np.float32)[None,:],count)
        return [int(i) for i in indices[0] if i>=0 and i not in excluded][:k]
