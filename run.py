#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from ldice.bedrock import BedrockClient
from ldice.executor import execute, extract_source, resolve_webppl
from ldice.game import validate
from ldice.oracle import exact
from ldice.prompts import build_prompt, reference_program
from ldice.scoring import score
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description='Single-bid WebPPL feasibility pilot')
    ap.add_argument('--mode',choices=['dry-run','reference','bedrock'],default='dry-run')
    ap.add_argument('--model',default='claude-haiku')
    ap.add_argument('--region',default=None)
    ap.add_argument('--cases',type=Path,default=ROOT/'data/cases.jsonl')
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--limit',type=int,default=None)
    ap.add_argument('--timeout',type=float,default=30)
    ap.add_argument('--api-timeout',type=float,default=120)
    ap.add_argument('--webppl',default=None)
    args=ap.parse_args()
    cases=[json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit<1: ap.error('--limit must be positive')
        cases=cases[:args.limit]
    for c in cases: validate(c)
    if args.mode!='dry-run': resolve_webppl(args.webppl)
    client=BedrockClient(args.model,args.region,args.api_timeout) if args.mode=='bedrock' else None
    out=args.out or ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    with (out/'results.jsonl').open('w') as log:
        for c in cases:
            start=time.perf_counter()
            prompt=build_prompt(c)
            (out/(c['id']+'.prompt.txt')).write_text(prompt)
            row={'case':c,'mode':args.mode,'model_id':client.model_id if client else None,
                 'region':client.region if client else None,'oracle':exact(c),'prompt':prompt}
            stage='generation'
            try:
                if client:
                    generated=client.generate(prompt);row['generation']=generated
                    source=extract_source(generated['text'])
                else: source=reference_program(c)
                row['source']=source
                (out/(c['id']+'.wppl')).write_text(source)
                if args.mode=='dry-run':
                    row['status']='dry_run_only'
                    row['execution_success']=None
                else:
                    stage='execution'
                    tick=time.perf_counter()
                    row['execution']=execute(source,args.timeout,args.webppl)
                    row['execution_latency_s']=time.perf_counter()-tick
                    row['execution_success']=True
                    row['scores']=score(row['execution']['probability'],row['oracle']['posterior'])
                    row['status']='pass' if row['scores']['numeric_pass'] else 'numeric_mismatch'
            except Exception as exc:
                row.update(status=stage+'_error',execution_success=False,error=str(exc),error_type=type(exc).__name__)
            row['total_latency_s']=time.perf_counter()-start
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();records.append(row)
            print(c['id'],row['status'])
    groups={}
    for condition in ('nonadversarial','adversarial'):
        rows=[r for r in records if r['case']['condition']==condition]
        scored=[r for r in rows if 'scores' in r]
        groups[condition]={'tasks':len(rows),'executed':len(scored),
            'passes':sum(r['scores']['numeric_pass'] for r in scored),
            'mean_tvd':sum(r['scores']['bernoulli_tvd'] for r in scored)/len(scored) if scored else None,
            'mean_regret':sum(r['scores']['regret'] for r in scored)/len(scored) if scored else None}
    summary={'mode':args.mode,'tasks':len(records),'groups':groups,
             'note':'Dry-run emits prompts/reference code and oracle only; no model or WebPPL execution.' if args.mode=='dry-run' else 'Reference mode validates hand-written programs; Bedrock mode evaluates model-generated programs.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Output:',out.resolve())
    return int(any(r['status'] not in ('pass','dry_run_only') for r in records))
if __name__=='__main__': raise SystemExit(main())
