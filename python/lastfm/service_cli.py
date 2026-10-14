"""Command line entry point for the shared recommendation service."""
import argparse
import json
from pathlib import Path
from .service import RecommendationService


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--bundle',type=Path,help='Load an explicit prepared bundle')
    parser.add_argument('--factors',type=int,default=64)
    parser.add_argument('--iterations',type=int,default=10)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    rec=sub.add_parser('recommend');rec.add_argument('user');rec.add_argument('--alpha',type=float)
    ev=sub.add_parser('evaluate');ev.add_argument('--max-users',type=int,default=20,help='0 means all holdout users')
    ev.add_argument('--test-end',default='2009-07-01T00:00:00Z')
    bench=sub.add_parser('benchmark');bench.add_argument('--max-users',type=int,default=10)
    for p in (rec,ev,bench):
        p.add_argument('--k',type=int,default=10);p.add_argument('--candidates',type=int,default=500)
        p.add_argument('--output',type=Path)
    for p in (rec,ev): p.add_argument('--retrieval',choices=('exact','hnsw'),default='exact')
    args=parser.parse_args()
    service=RecommendationService.load(args.bundle) if args.bundle else RecommendationService.open(args.root,factors=args.factors,iterations=args.iterations)
    if args.command=='prepare': result=dict(bundle=str(service.folder),service_key=service.key,fit_seconds=service.fit_seconds)
    elif args.command=='recommend': result=service.recommend(args.user,k=args.k,alpha=args.alpha,retrieval=args.retrieval,candidates=args.candidates)
    else:
        from .service_evaluation import evaluate_service,benchmark_service
        if args.max_users<0: parser.error('--max-users must be nonnegative')
        if args.command=='evaluate': result=evaluate_service(service,k_list=[args.k],test_end=args.test_end,max_users=args.max_users or None,retrieval=args.retrieval,candidates=args.candidates)
        else: result=benchmark_service(service,users=service.users[:args.max_users] if args.max_users else service.users,k=args.k,candidates=args.candidates)
    output=getattr(args,'output',None)
    if output:
        from .service_evaluation import write_report
        write_report(result,output)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
