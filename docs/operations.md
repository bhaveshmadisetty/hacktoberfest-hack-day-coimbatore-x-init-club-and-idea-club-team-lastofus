# Operations

## Running it

```bash
pip install -r requirements.txt
cd web && npm install && cd ..
cp .env.example .env          # add GEMMA_API_KEY

# terminal 1
PYTHONPATH=src python -m uvicorn arbiter.api:app --reload --port 8000
# terminal 2
cd web && npm run dev
```

Dashboard at <http://localhost:3000>, API at <http://localhost:8000>.

## Environment

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMMA_API_KEY` | **yes** | — | Primary key. Without it Arbiter runs degraded and refuses to report metrics |
| `GEMMA_API_KEY_2..5` | no | — | Additional keys for rotation |
| `GEMMA_MODEL` | no | `gemma-4-26b-a4b-it` | `gemma-4-31b-it` reasons better and costs more |
| `ARBITER_DB` | no | `./arbiter.db` | SQLite path |
| `ARBITER_KEY_SALT` | no | dev default | **Change in any shared deployment.** Rotating it invalidates every issued API key |
| `ARBITER_PUBLIC_URL` | no | `http://localhost:8000` | URL shown in the dashboard's connect snippet |
| `ARBITER_CACHE` | no | `./demo_cache.json` | Response cache for offline replay |
| `ARBITER_CORS_ORIGINS` | no | `http://localhost:3000` | Allowed dashboard origins |
| `ARBITER_ALLOW_VERCEL` | no | unset | Allow any `*.vercel.app` preview origin |

`.env` is gitignored. Verify nothing leaked:

```bash
git log --all --name-only | grep -c "^\.env$"      # must print 0
```

## API key rotation

A single free-tier Gemma key runs out of quota quickly, and when it does every
narration in the dashboard degrades to fallback output at once. Arbiter spreads
calls across every key configured.

```
GEMMA_API_KEY=AIza...        slot 1
GEMMA_API_KEY_2=AIza...      slot 2
GEMMA_API_KEY_3=             (skipped — blank slots are not counted)
```

- **Round robin.** Each call takes the next available key.
- **Failover on 429.** A rate-limited key is parked for 60 seconds and the call
  is immediately retried on the next key, once per configured key.
- **Non-quota errors do not park a key** — a malformed request is not the key's
  fault, and parking would remove a working credential.
- **When every key is cooling down**, the one closest to recovery is tried
  anyway: a likely-429 attempt beats degrading without trying.

Pool health is in `/health` and contains no key material:

```json
{ "key_pool": { "configured_keys": 3, "available_now": 2,
                "keys": [{ "slot": 1, "calls": 41, "rate_limited": 1,
                           "cooling_down": true, "cooldown_remaining_s": 38.4 }] } }
```

## Failure modes

| Symptom | Cause | What happens |
|---|---|---|
| Verdicts say `source: heuristic` | No usable key, or all exhausted | Offline fallback runs; metrics report `reportable: false` and are not quotable |
| "Registering not working" | Email already registered | The API returns a clear message; the UI offers to switch to Sign in |
| Model appears with "none" features | SDK sent no feature names | Narrations cannot cite features. Pass `features=` or a `feature_extractor` |
| Telemetry arrives, no events | Threshold not crossed, or fewer than 25 drifted samples | By design — reasoning on a 1-row drifted window produces a wrong narration |
| Dashboard shows "Cannot reach the API" | Backend down, or CORS | Check the API is on :8000 and `ARBITER_CORS_ORIGINS` includes the dashboard origin |

## Verification

```bash
python -m pytest                                # full suite
PYTHONPATH=src python -m arbiter.run_replay     # the README's measured numbers
PYTHONPATH=src python -m arbiter.sdk_demo       # live SDK -> Gemma proof
ARBITER_DB=./e2e.db python scripts/e2e_flow.py  # account -> key -> 3 models -> verdicts
cd web && npx next build                        # type-check and build
```

`e2e_flow.py` takes credentials from `ARBITER_TEST_EMAIL` and
`ARBITER_TEST_PASSWORD` — a password in version control is a committed
credential even for a local test account.

## Security notes

| Credential | Storage | Recoverable? |
|---|---|---|
| Password | PBKDF2-HMAC-SHA256, 600k iterations, per-user salt | No |
| Session token | SHA-256, 14-day expiry | No |
| API key | Salted SHA-256 | No — shown once at creation |

PBKDF2 is a deliberate scope decision: it avoids a native dependency for a
hackathon build while never storing a recoverable password. **A production
deployment should move to Argon2id**, which is memory-hard and therefore harder
to attack with GPUs.

Changing an email requires the current password, so a stolen session token
alone cannot redirect future password resets.
