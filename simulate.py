#!/usr/bin/env python3
"""Optional reproducible generative sanity check; no API or WebPPL dependency."""
import argparse
import json
import random
from pathlib import Path
from ldice.game import simulate
from ldice.oracle import exact
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.samples<1: ap.error('--samples must be positive')
    rng=random.Random(args.seed)
    for line in (ROOT/'data/cases.jsonl').read_text().splitlines():
        c=json.loads(line)
        rows=[simulate(c,rng) for _ in range(args.samples)]
        bids=[r for r in rows if r['bid_emitted']]
        print(json.dumps({'id':c['id'],'samples':args.samples,'seed':args.seed,
              'emitted':len(bids),'empirical_prior':sum(r['claim_true'] for r in rows)/len(rows),
              'empirical_posterior':sum(r['claim_true'] for r in bids)/len(bids) if bids else None,
              'oracle':exact(c)}))
