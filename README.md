# Minimal Liar's Dice → WebPPL feasibility prototype

A runnable, single-bid inference pilot for a WebPPL final project. It contains **8 matched pairs (16 tasks)**, a generative dice simulator, an independent exact binomial oracle, prompt builder, Python boto3 Converse client, local WebPPL executor, scoring, and JSONL logs. Python 3.10+ and Node.js 18+ are required for full execution. Python 3.11 is recommended and has been verified locally; it avoids the pip bootstrap failure observed with the local Python 3.15 installation. Python's standard library alone runs the dry-run and offline unit tests.

## Quick start: no API calls

```bash
unzip liars-dice-webppl.zip
cd liars-dice-webppl
python3 -m unittest discover -s tests -v
python3 run.py --mode dry-run --out runs/dry
```

Before WebPPL is installed, the integration test is explicitly skipped. Dry-run writes prompts, hand-written reference programs, and oracle values; it **does not** execute WebPPL, call Bedrock, or produce model accuracy scores. A dry-run status is `dry_run_only`, never `pass`.

To execute the actual probabilistic programs locally, still without API calls:

```bash
npm ci --ignore-scripts --no-audit --no-fund
python3 -m unittest discover -s tests -v
python3 run.py --mode reference --out runs/reference
python3 simulate.py --samples 20000 --seed 42 > runs/simulation.jsonl
```

The WebPPL integration test enumerates concrete hidden dice for every task and compares its output to the independently implemented binomial oracle. `examples/` contains the verified reference-run JSONL, summary, oracle table, and test output. These are infrastructure checks of hand-written reference programs, **not LLM results**.

## Rules and consistent bid policy

Two hands: the observer's dice are known; the opponent's dice are hidden. Hidden dice are iid fair d6. Ones are **not wild**. The claim is “at least q dice show face f” across both hands. This is a simplified inference scenario inspired by Liar's Dice; it does not implement a full bidding game, raises, elimination, or optimal play.

A **single fixed candidate bid is chosen before the roll**. After the hidden dice are rolled once, an omniscient bidder who knows both hands decides whether to emit that candidate bid. This deliberately stylized knowledge assumption makes truth-dependent emission unambiguous. It differs from real Liar's Dice, where a player normally sees only their own hand. There is no normalized choice among multiple bids.

| Condition | P(emit bid \| claim true) | P(emit bid \| claim false) |
|---|---:|---:|
| nonadversarial | 0.5 | 0.5 |
| adversarial | 1 | 0.3 |

All 16 tasks condition on **bid emitted = true**. An emitted claim may be false; do not condition the claim itself to be true. “Adversarial” here means a known state-dependent bluffing signal policy, not an equilibrium or a claim that this policy is optimal. Each pair has identical dice, face, and quantity; only the policy changes. Unemitted bids can be simulated but are outside this pilot's inference dataset.

Let `k` be known matches, `n` hidden dice, and `X ~ Binomial(n,1/6)`. For `T = [X+k >= q]`:

```text
p = P(T) = sum_{x+k >= q} choose(n,x) (1/6)^x (5/6)^(n-x)
P(T | emitted) = p*a / (p*a + (1-p)*b)
```

The oracle uses Python `Fraction` and `math.comb`. It shares neither simulation code nor WebPPL inference code.

**Base example:** own `[2,6]`, three hidden dice, claim at least three sixes. At least two of the three hidden dice must be six:

```text
prior = 3*(1/6)^2*(5/6) + (1/6)^3 = 2/27 = 0.074074074074
nonadversarial posterior = 2/27
adversarial posterior = (2/27) / [(2/27) + 0.3*(25/27)]
                      = 4/19 = 0.210526315789
```

## Dataset

`data/cases.jsonl` is deterministic and contains no oracle answers. Prompts are built only from each case's public rules and observations. Oracle values are added to evaluation logs, never model prompts.

| Pair | Observer dice | Hidden | Face | At least | Focus |
|---|---|---:|---:|---:|---|
| P01 | [2,6] | 3 | 6 | 3 | Base regression |
| P02 | [1,2] | 3 | 6 | 1 | Optimal decision changes between conditions |
| P03 | [6,6] | 2 | 6 | 3 | Known matches |
| P04 | [2,4] | 4 | 2 | 3 | More hidden dice |
| P05 | [1,3] | 3 | 1 | 2 | Ones count only as ones |
| P06 | [4,5] | 2 | 3 | 2 | No known matches |
| P07 | [6,6] | 2 | 6 | 2 | Claim already certain |
| P08 | [1,2] | 2 | 6 | 5 | Impossible claim, but bid still observable |

The two boundary pairs check implementation, not difficulty. Eight small pairs establish feasibility, not statistical evidence of general model capability. Repeat real runs and expand tasks before drawing research conclusions.

## Bedrock setup and model runs

The client reuses the user's earlier `scallop-semantic-mini-pilot-v2/model.py` approach: `boto3.client('bedrock-runtime')`, `.converse(...)`, user text content blocks, extraction of text blocks, temperature `0.5`, and max output tokens `6400`. Exact aliases and default region are preserved:

| CLI alias | Bedrock model / inference profile ID |
|---|---|
| claude-haiku | us.anthropic.claude-haiku-4-5-20251001-v1:0 |
| claude-sonnet | us.anthropic.claude-sonnet-4-6 |

Default region is `us-east-2`, overridden by `--region`, `AWS_REGION`, or `AWS_DEFAULT_REGION` (in that order). An arbitrary profile/model ID can be passed directly with `--model`. Unlike the old wrapper, the client also allows boto3's normal credential chain, configures SDK connection/read timeouts, logs token usage, and rejects empty/truncated outputs. Extended/adaptive thinking is not enabled in this minimal baseline.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
```

Authenticate with your existing Bedrock bearer-token setup (same variable as before), or an existing AWS profile/role. Enter a token without placing it in shell history:

```bash
read -s AWS_BEARER_TOKEN_BEDROCK
export AWS_BEARER_TOKEN_BEDROCK
export AWS_REGION=us-east-2
```

Enter your token after the `read` command and press Return. For a standard AWS profile instead, use `export AWS_PROFILE=your-profile` with its configured credentials. No credentials belong in project files or logs.

Start with one task (one paid request), then run each full 16-task condition set:

```bash
python run.py --mode bedrock --model claude-haiku --limit 1 --out runs/haiku-smoke
python run.py --mode bedrock --model claude-haiku --out runs/haiku
python run.py --mode bedrock --model claude-sonnet --out runs/sonnet
```

Each full model run makes 16 requests with no application-level retries or repairs. The SDK is configured for one attempt. Two full model runs make 32 requests, in addition to the smoke request. Model access, credentials, quotas, and inference-profile permissions must work in your AWS account. Cross-region inference may require access to destination-region resources. `--api-timeout 120` sets the SDK read timeout (not a strict total wall-clock deadline); `--timeout 30` bounds each WebPPL subprocess. A missing WebPPL binary is detected before any model requests.

**Bedrock was subsequently tested with Claude Haiku 4.5:** the first full run executed all 16 programs and passed 15/16 numeric checks (nonadversarial 8/8, adversarial 7/8). The P02 adversarial program used native JavaScript `.some(callback)` instead of WebPPL `any(...)`, yielding 1.0 rather than the oracle posterior 0.7081712062. Replacing that expression locally recovered the oracle result; the original experiment remains scored as a failure. The likelihood policy was correctly generated. All decisions in that run had zero regret, illustrating why probability fidelity and decision quality are separate metrics. This is one pilot run, not a statistically established condition effect. Sonnet execution has not been verified. Mocked Converse contract tests alone cannot establish account access or real model behavior. The project logs failures per case and returns a nonzero process status on failed execution or numeric mismatch. Failures are not silently scored as correct.

## Outputs and evaluation

Each output directory has `results.jsonl`, `summary.json`, and per-case `.prompt.txt` / `.wppl` files. A result includes case configuration, mode, model ID/region, full prompt, exact oracle values, generated source, model response text, usage/stop reason, latency, subprocess output, status, and scores where available. Generation and execution errors have separate statuses. Use a fresh `--out` directory per run; reusing one overwrites its logs.

For a Boolean claim, total variation distance equals absolute posterior error. Numeric pass tolerance is `1e-6`. Invalid, nonfinite, out-of-range, missing, or duplicate result records fail parsing. The required WebPPL output line is:

```text
RESULT_JSON:{"probability":0.21052631578947367}
```

The model must infer a Boolean distribution using exact enumeration, then print `Math.exp(dist.score(true))`.

For a simple decision metric, `accept` earns +1 if the claim is true and -1 otherwise; `challenge` has the opposite payoff. Choose accept for posterior ≥ 0.5 (tie rule). Regret is the oracle-optimal expected reward minus the chosen action's expected reward. These are benchmark decision utilities, not full-game Liar's Dice rewards. At posterior 0.5, both actions have zero regret, although the tie rule labels accept as the canonical correct action.

Summary reports per-condition task count, executed/scored count, numeric passes, mean TVD, and mean regret. Interpret pass counts relative to all tasks, and inspect failures: means include only scored tasks. Pair IDs support matched comparison. Numeric agreement alone does not prove structural modeling fidelity; inspect generated code for hardcoded answers or incorrect models that coincidentally match a test case.

The executor runs local generated code with a subprocess timeout; it is **not a security sandbox**. Run generated programs in an isolated environment if they are untrusted. Local `node_modules/.bin/webppl` is preferred; `--webppl /absolute/path/to/webppl` overrides it, with PATH fallback when no override is given.

## Files and verification

- `ldice/game.py`: dice simulator and input validation.
- `ldice/oracle.py`: independent exact binomial inference.
- `ldice/prompts.py`: prompt builder and hand-written reference WebPPL.
- `ldice/bedrock.py`: lazy-imported boto3 Converse client.
- `ldice/executor.py`: code-fence extraction, WebPPL subprocess and result parsing.
- `ldice/scoring.py`: posterior error, Bernoulli TVD, numeric pass, decision and regret.
- `run.py`: dry-run / reference / Bedrock experiment pipeline.
- `simulate.py`: seeded generative sanity check, emits JSONL.
- `tests/test_prototype.py`: offline unit and local WebPPL integration tests.
- `scripts/package.py`: clean ZIP builder (excludes dependencies, credentials, caches, runtime logs).
- `VERIFICATION.md`: exact checks performed for delivery.

Official references: [boto3 Converse API](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/converse.html), [Haiku 4.5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html), [Sonnet 4.6](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html), and [WebPPL inference](https://docs.webppl.org/en/master/inference/methods.html).
