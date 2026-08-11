# Advanced notes

The hero README keeps the path to a first deploy as short as possible. This file
collects everything intentionally left out of it.

## The webhook verifier is vendored

`src/explaining_markets/webhook_verification.py` is a verbatim copy of the
competition's reference verifier (stdlib-only, zero runtime dependencies). It's
vendored — not installed from PyPI — so the starter is self-contained.

If the competition later publishes the verifier as a package, you can delete the
vendored file and depend on the pinned package instead, importing
`verify_webhook` / `WebhookVerificationError` from there. The frozen test vectors
in `tests/test_vectors.json` will keep passing either way, since both
implementations pin to the same values.

## Credentials: `.env` vs Modal's secret store

By default the app reads credentials from your local `.env` at deploy time via
`modal.Secret.from_dotenv(__file__)` in `modal_app.py`. That keeps setup to "fill
in a file and deploy" — no secret-management command to copy.

If you'd rather keep credentials in Modal's secret store (e.g. for CI, or to avoid
a local file), create a named secret once:

```bash
uv run modal secret create explaining-markets \
  EM_API_KEY=... EM_WEBHOOK_SECRET=whsec_... OPENAI_API_KEY=...
```

then swap the decorator in `modal_app.py`:

```python
secrets=[modal.Secret.from_name("explaining-markets")]
```

## Webhook path and URL

The handler is registered on both `POST /` (primary) and `POST /competition/webhook`
(alias), so either URL works. We serve it at the root so the URL Modal prints on
deploy *is* your webhook URL, with nothing to append.

The URL's subdomain comes from `@modal.asgi_app(label="explaining-markets")`, which
makes it `https://{your-workspace}--explaining-markets.modal.run` instead of the
longer default. Change the `label` to change the subdomain.

## Two clocks: 20 seconds to ACK, 5 minutes to predict

`modal_app.py` verifies → ACKs 200 → spawns `predict_and_submit`. The platform
runs two independent timers:

| Clock | Budget | Starts | Miss it and… |
|---|---|---|---|
| Delivery ACK | 20 s | when the platform POSTs to you | the delivery is retried up to 5 times over ~30 min; 5 consecutive failures emails your admins, ~50 disables your webhook |
| Prediction window | 5 min | when you ACK 200 | your prediction is tagged late and dropped at scoring |

The 5-minute window only opens once you ACK. Predicting before the ACK spends it
inside the 20-second budget — a 25-second model call is fine against your
prediction window and a hard failure against your delivery budget.

The work runs as a spawned Modal function with its own container, so it survives
the web container scaling down. Once you ACK, the platform will not redeliver, so
Modal's function retries (and the single retry on the model call in `predict.py`)
are your durability layer. `modal.Queue` is for fan-out or rate-limiting, not
durability.

## Idempotency

Deliveries are deduped on the `Webhook-Id` header (equal to `event["id"]`), which
is stable across retries. The guard has three states:

* **in flight** — claimed on arrival, before any work. A duplicate landing while
  the first job runs is skipped, so you never pay for the same model call twice.
* **done** — set only after the API accepts your prediction. Permanent.
* **released** — if the job raises, the claim is dropped, so the next delivery of
  that `Webhook-Id` re-runs it.

Marking an event done up front would be the bug: a failed prediction would look
handled.

A replay of an older event arrives with a fresh `Webhook-Id`, so a "done" marker
never blocks one.

The store is a `modal.Dict`, which persists across redeploys. The claim uses
`put(..., skip_if_exists=True)` so it is atomic: two containers handling a
duplicate delivery at the same moment can't both win.

## Not included by design

No Docker, Terraform, GitHub Actions, custom CLI, or multiple deployment modes —
Modal's persistent deployments, public web endpoints, and Secrets cover the
starter. Add those only if your own setup needs them.
