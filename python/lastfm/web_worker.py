"""Isolated offline run worker, with atomic result publication."""
import json
from pathlib import Path
import sys
from .recommend_engine import get_recommendations_with_explanations

if __name__=='__main__':
    root,folder=map(Path,sys.argv[1:3])
    options=json.loads((folder/'request.json').read_text())
    print('Preparing chronological matrices and fitting the collaborative model.',flush=True)
    print('This offline stage can take several minutes. No percentage estimate is available.',flush=True)
    result=get_recommendations_with_explanations(root,user_query=options.pop('user'),**options)
    temporary=folder/'result.partial'
    temporary.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    temporary.replace(folder/'result.json')
    print('Recommendation mix and explanations are ready.',flush=True)
