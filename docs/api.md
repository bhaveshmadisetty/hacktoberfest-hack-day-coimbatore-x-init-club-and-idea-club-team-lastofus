# HTTP API

Base URL `http://localhost:8000`. Interactive reference at `/docs`.

Two authentication schemes, because there are two kinds of caller:

| Caller | Header | Used for |
|---|---|---|
| Dashboard (a person) | `Authorization: Bearer <session token>` | `/auth/*`, `/api/*` |
| Your model (a machine) | `X-API-Key: ak_live_…` | `/models/*`, `/predict/*`, `/retrain/*` |

## Auth

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/register` | `{email, password, name?}` → session. 400 if the email is taken or the password is under 8 characters |
| POST | `/auth/login` | `{email, password}` → session. 401 on bad credentials — the same message either way, so it does not reveal which emails are registered |
| POST | `/auth/logout` | Ends the current session |
| GET | `/auth/me` | The signed-in user |
| PATCH | `/auth/email` | `{new_email, password}`. **Current password required** |
| PATCH | `/auth/password` | `{current_password, new_password}`. Invalidates every session |

## Dashboard API

All session-authenticated and scoped to the caller's account.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Status, Gemma availability, **key-pool health** |
| POST | `/api/keys` | A new API key. **The plaintext appears only in this response** |
| GET | `/api/keys` | Key hints and usage. Never plaintext |
| DELETE | `/api/keys/{id}` | Revokes a key |
| GET | `/api/models` | Your models with telemetry counts and latest verdict |
| GET | `/api/models/{id}/metrics` | That model's metric row and drift series |
| GET | `/api/models/{id}/events` | Its reasoned drift events |
| GET | `/api/models/{id}/telemetry` | Raw telemetry |
| GET | `/api/models/{id}/docs` | **Gemma-written incident report** (Markdown) |
| POST | `/api/models/{id}/chat` | `{question, history?}` — scoped to that model |
| POST | `/api/chat` | `{question, history?}` — across every model you own |
| GET | `/api/live/events` | Drift events across your models |
| GET | `/api/live/events/{id}` | One event with per-feature detail |

### Demo corpus (shared, synthetic, read-only)

`/api/metrics`, `/api/events`, `/api/events/{id}`, `/api/ask`,
`/api/replay/reset`, `/api/digest`. The 20-event seeded replay used to produce
the README's measured numbers. Deliberately separate from live telemetry so a
scripted figure is never mistaken for a measured one.

## SDK wire protocol

Implemented from `driftguard-ai-sdk` 1.0.4's actual source. Point the SDK at
Arbiter with two environment variables; the model code does not change.

| Method | Path | SDK source | Purpose |
|---|---|---|---|
| POST | `/models/register` | `tracker.py:132` | Registers a model **with its ordered feature names** |
| POST | `/predict/{model_id}` | `tracker.py:328` | Per-prediction telemetry |
| GET | `/models/{model_id}` | `tracker.py:85` | Version lookup |
| POST | `/retrain/{model_id}` | `callback_runner.py:266` | The gate — returns Arbiter's verdict |
| POST | `/retrain/{model_id}/complete` | `callback_runner.py:302` | Completion callback |

Registration is not bookkeeping: the `features` array is what turns a positional
drift score into a feature name, and a model registered without it gets a
warning in the response and a flag in the dashboard.

`/predict` returns in roughly 40ms and reports `reasoning_queued`. The verdict
arrives later via `/api/live/events` — the SDK's telemetry worker has a 5-second
timeout, and reasoning takes about 25 seconds.

### Authentication behaviour

Until the first API key is minted, the SDK routes accept unauthenticated
telemetry so a new user can see the product work before learning the key flow.
Once any key exists, authentication is enforced — otherwise minting one would
silently weaken the thing it was meant to secure.

The SDK also places `api_key` in the request body. Arbiter ignores it: the
header is the only trusted source.

## Example

```bash
# 1 · sign in
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"your-password"}' | jq -r .token)

# 2 · mint a key
KEY=$(curl -s -X POST localhost:8000/api/keys \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"label":"laptop"}' | jq -r .key)

# 3 · register a model, with names
curl -s -X POST localhost:8000/models/register \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"my-model","features":["amount","category"],"drift_threshold":0.15}'

# 4 · send telemetry
curl -s -X POST localhost:8000/predict/my-model \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"features":[240.0,8.4],"prediction":[0],"drift_score":0.04}'

# 5 · read the verdict once reasoning completes
curl -s localhost:8000/api/live/events -H "Authorization: Bearer $TOKEN"
```
