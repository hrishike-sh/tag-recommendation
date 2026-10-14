"""Isolated offline run worker, with atomic result publication."""
import json
from pathlib import Path
import sys
from .service import RecommendationService

if __name__=='__main__':
    root,folder=map(Path,sys.argv[1:3])
    options=json.loads((folder/'request.json').read_text())
    print('Loading the shared service bundle; first use fits and saves the model.',flush=True)
    service=RecommendationService.open(root)
    print(f'Service {service.key} ready. Both streams use the same snapshot.',flush=True)
    kind=options.pop('kind','recommend')
    user=options.pop('user'); max_users=options.pop('max_users',20)
    if kind=='recommend':
        result=service.recommend(user,**options)
    else:
        from .service_evaluation import evaluate_service,benchmark_service
        if kind=='evaluation':
            options['k_list']=[options.pop('k')]
            result=evaluate_service(service,max_users=max_users,**options)
        else:
            result=benchmark_service(service,users=service.users[:max_users],
                k=options['k'],candidates=options['candidates'])
    temporary=folder/'result.partial'
    temporary.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    temporary.replace(folder/'result.json')
    print('Run completed; results are ready.',flush=True)
