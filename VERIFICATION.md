# Delivery verification — 2026-11-07

The initial delivery checks below were performed without real Bedrock/model API calls. Subsequent live validation is documented at the end.

- Python 3.15.2, Node.js v18.20.8, npm 10.8.2, WebPPL 0.9.15 on macOS.
- `npm ci --offline --cache /private/tmp/liars-dice-npm-cache --ignore-scripts --no-audit --no-fund`: succeeded using the dependency cache after initial npm installation. `package-lock.json` pins the dependency tree.
- `python3 -m unittest discover -s tests -v`: **13 tests passed**, including one integration test covering all **16** reference WebPPL programs. See `examples/tests.txt`.
- `python3 run.py --mode dry-run`: **16 dry-run records**, no numeric passes claimed, no SDK client initialized or WebPPL invoked.
- `python3 run.py --mode reference`: **16/16 numeric passes** versus exact oracle; per-condition mean TVD below 1e-15, regret zero.
- Seeded generative simulator: 20,000 samples per case, seed 42. `examples/simulation.jsonl` contains empirical frequencies; these naturally differ from exact values.
- Base regression: prior `2/27`, adversarial posterior `4/19`; independent brute-force concrete dice and binomial oracle agree across all cases.
- Mocked Converse tests check both model IDs, request shape, text extraction, usage metadata, empty response and truncation handling. Mocked executor tests cover timeout and process failure; parser rejects invalid/nonfinite probabilities and duplicate result lines.
- ZIP rebuilt with allowlisted files; CRC/member checks and an extracted-project dry-run verified. No node_modules, .venv, .git, .env, credentials, caches, or unrelated user files are included.

## Prior-code provenance

The actual local source `scallop-semantic-mini-pilot-v2/model.py` was read. Its SHA-256 is `57934c909d09eeb4d8a3871a27ff28834f2bde13b17dcc91ebf34054af2ed259`. The two exact model aliases, default `us-east-2`, boto3 runtime client, Converse request style, temperature 0.5, maxTokens 6400, and text-block extraction were reused. The old wrapper itself and any credentials were not packaged. User discussion context was read from the referenced chat.

## Limits of initial validation

No real Bedrock inference was invoked during the initial delivery checks. AWS credentials, account permissions, regional routing/access, quotas, billing, and actual generated-model quality remain untested. Thinking mode is not enabled. Tested local WebPPL reference results establish feasibility of the execution/oracle pipeline, not LLM synthesis accuracy. Python 3.10+ compatibility is intended but only Python 3.15.2 was exercised here.

## Subsequent local and live validation

- Python 3.11.3 environment: all 13 tests passed, including 16-case WebPPL integration. boto3 installed successfully.
- User-run Claude Haiku 4.5 Bedrock experiment: 16/16 programs executed, 15/16 numeric passes; nonadversarial 8/8, adversarial 7/8.
- P02 adversarial predicted 1.0; oracle 182/257 = 0.708171206225681. Generated code used native `.some(callback)`. A local diagnostic replacing only that expression with WebPPL `any(...)` produced 0.7081712062256811. Original run and baseline scoring were preserved.
- Sonnet live behavior remains unverified. Raw local runs and virtual environments are excluded from Git.
