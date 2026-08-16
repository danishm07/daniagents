"""Modal deployment for the Explaining Markets starter.

This is plumbing — you shouldn't need to edit it. It defines a small FastAPI app
and deploys it as a persistent, public web endpoint:

    GET  /    health check
    POST /    receive a signed event, verify, ACK, then predict and submit
              (POST /competition/webhook is kept as an alias of the same handler)

The webhook is served at the root path on purpose: the URL Modal prints on deploy
*is* your webhook URL — paste it into the portal as-is, nothing to append.

Deploy:    uv run modal deploy modal_app.py
Dev/local: uv run modal serve modal_app.py

The webhook handler ACKs first, then predicts. It verifies the signature, returns
200, and spawns `predict_and_submit` — a separate Modal function with its own
container — to run your `predict()` from predict.py and POST the result. Two
clocks:

  * 20 seconds to ACK the delivery. Miss it and the platform retries; repeated
    failures disable your webhook.
  * 5 minutes from that ACK to submit your prediction.

Predicting before the ACK spends the 5-minute budget inside the 20-second one.
Spawning rather than using a background task also means the work doesn't depend
on the web container staying alive.

Deliveries are deduped on the `Webhook-Id` header (the server retries on
4xx/5xx/timeout, so the same event can arrive more than once).

Note: we deliberately do NOT use `from __future__ import annotations` here. The
route handlers are defined inside `web()`, and FastAPI must see the real `Request`
/ `Response` classes (not stringized annotations it can't resolve from this nested
scope) to inject them correctly — otherwise it treats `request` as a query
parameter and rejects every delivery with 422.
"""

import modal

app = modal.App("explaining-markets-starter")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "openai", "pydantic")
    .add_local_python_source("explaining_markets", "predict")
)

# Distributed key-value store for idempotency, keyed on the Webhook-Id header.
# Three states:
#
#   "in_flight"   a job is running right now — skip duplicates so you never pay
#                 for the same model call twice
#   "done"        the API accepted a prediction — skip forever
#   absent        never seen, or the last attempt raised — (re)run it
#
# Marking an event done up front would be the bug: a failed prediction would
# look handled. This Dict persists across redeploys, so "done" is durable.
seen_webhooks = modal.Dict.from_name("em-webhook-dedupe", create_if_missing=True)

# Every prediction we submit, keyed by event_id, with the rung of the
# degradation ladder that produced it. Modal's log retention is a rolling
# window of minutes, so without this there is no way to answer "how many events
# have we actually covered, and how many were neutral fallbacks?" — the two
# questions that decide our rank, since the contest metric mean-fills anything
# we miss.
prediction_log = modal.Dict.from_name("em-prediction-log", create_if_missing=True)

#: Submission retries. Duplicates are accepted with 201 and only the first
#: prediction per event is ever scored, so re-POSTing costs nothing and cannot
#: overwrite a good submission. Losing a computed prediction to a single failed
#: POST is the most expensive failure mode available to us.
SUBMIT_ATTEMPTS = 4
SUBMIT_BACKOFF_SECONDS = 2.0

# Credentials are read from your local .env at deploy time (see .env.example).
# Prefer Modal's secret store instead? See docs/advanced.md.
secrets = [modal.Secret.from_dotenv(__file__)]


def _claim(webhook_id):
    """Reserve this webhook_id. False means it's already in flight or done.

    `skip_if_exists` makes this an atomic claim, so two containers handling a
    duplicate delivery at the same moment can't both win.
    """
    if not webhook_id:
        return True
    return seen_webhooks.put(webhook_id, "in_flight", skip_if_exists=True)


async def _claim_aio(webhook_id):
    """`_claim` for the async route.

    Modal's blocking interfaces run their own event loop under the hood, so
    calling them from inside an `async def` stalls the loop — the exact problem
    ACKing first is meant to solve. The `.aio` variants are the async-native
    ones; the request path must use these, and only these.
    """
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(webhook_id, "in_flight", skip_if_exists=True)


def _release(webhook_id, submitted):
    """Mark the claim done on success, or drop it so a redelivery can retry."""
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


def _usable(predictions) -> bool:
    """Would the scorer accept this? Out-of-range counts as a missed event.

    Checked before submitting rather than trusted, because a malformed
    prediction and no prediction score the same, and this is the last place we
    can still substitute something valid.
    """
    if not isinstance(predictions, list) or not predictions:
        return False
    for row in predictions:
        if not isinstance(row, dict):
            return False
        value = row.get("predicted_percentile")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not (0.0 <= float(value) <= 1.0):
            return False
        if not isinstance(row.get("identifier_value"), str):
            return False
    return True


def _predict_with_ladder(event: dict):
    """``(predictions, rung, detail)`` — full pipeline, one retry, then neutral.

    The ladder CLAUDE.md has always specified and this file never implemented:
    a neutral 0.5 scores zero, but **no prediction scores worse**, because the
    contest metric mean-fills every event we miss. Until now any exception here
    submitted nothing at all, and nothing upstream retries a delivery we have
    already ACKed.

    The neutral rung deliberately depends on nothing but the webhook body:
    ``neutral_predictions`` reads ``focal_assets``, so it is reachable even when
    the ``information_url`` fetch is the thing that failed — which is the most
    likely failure, since it is a network call with a 15s timeout.
    """
    from explaining_markets.event_utils import is_test, neutral_predictions
    from predict import predict

    event_id = event.get("event_id")
    if is_test(event):
        return neutral_predictions(event), "test", None

    for rung in ("full", "retry"):
        try:
            predictions = predict(event)
        except Exception as exc:
            print(f"[DEGRADE] {event_id} rung={rung} failed: {type(exc).__name__}: {exc}")
            continue
        if _usable(predictions):
            return predictions, rung, None
        print(f"[DEGRADE] {event_id} rung={rung} returned unusable output: {predictions!r}")

    print(f"[DEGRADE] {event_id} rung=neutral — submitting 0.5 rather than nothing")
    return neutral_predictions(event), "neutral", "predict() failed twice"


def _submit_with_retries(event_id: str, predictions: list, config) -> int:
    """POST until it sticks. Returns the attempt that succeeded.

    Retrying is free: duplicates return 201, only the first prediction per event
    is scored, and a re-POST cannot overwrite a good one. Before this, a single
    transient POST failure discarded a prediction we had already paid for.
    """
    import time

    from explaining_markets.client import submit_predictions

    last = None
    for attempt in range(1, SUBMIT_ATTEMPTS + 1):
        try:
            submit_predictions(event_id=event_id, predictions=predictions, config=config)
            return attempt
        except Exception as exc:
            last = exc
            print(f"[SUBMIT] {event_id} attempt {attempt}/{SUBMIT_ATTEMPTS} failed: "
                  f"{type(exc).__name__}: {exc}")
            if attempt < SUBMIT_ATTEMPTS:
                time.sleep(SUBMIT_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise last


@app.function(image=image, secrets=secrets, timeout=600, retries=0)
def predict_and_submit(event: dict, webhook_id: str | None = None):
    """Run the model and submit the prediction, off the request path.

    Runs in its own container, so it is unaffected by the web endpoint scaling
    down. The delivery has already been ACKed by the time this starts, which
    means nothing upstream will retry it — so every failure has to be handled
    here or the event is lost permanently.
    """
    import time

    from explaining_markets.config import Config

    event_id = event.get("event_id")
    submitted = False
    predictions, rung, detail = _predict_with_ladder(event)

    try:
        attempts = _submit_with_retries(event_id, predictions, Config.from_env())
        submitted = True
    except Exception as exc:
        attempts = SUBMIT_ATTEMPTS
        detail = f"submit failed after {SUBMIT_ATTEMPTS} attempts: {exc}"
        print(f"[ERROR] {event_id} not submitted — {detail}")

    for row in predictions:
        print(
            # [PREDICT] stays in predict.py (it reports the fact count); this is
            # the outcome line, and it is the one that says whether the event was
            # actually covered and on which rung.
            f"[OUTCOME] event={event_id} ticker={row['identifier_value']} "
            f"p={row['predicted_percentile']:.3f} rung={rung} "
            f"submitted={submitted} attempts={attempts}"
        )

    try:
        prediction_log[event_id] = {
            "event_id": event_id,
            "submitted_at": time.time(),
            "rung": rung,
            "submitted": submitted,
            "attempts": attempts,
            "detail": detail,
            "predictions": predictions,
        }
    except Exception as exc:  # bookkeeping must never cost a submission
        print(f"[WARN] {event_id} prediction_log write failed: {exc}")

    _release(webhook_id, submitted)


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label="explaining-markets")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import Config
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets starter")

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": "explaining-markets-starter"}

    @api.post("/")
    @api.post("/competition/webhook")  # alias, so an explicit-path URL also works
    async def competition_webhook(request: Request) -> Response:
        config = Config.from_env()

        raw_body = await request.body()  # raw bytes — never request.json()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(content=str(exc), status_code=401)

        webhook_id = event.get("id")
        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)
        # Everything slow happens after this 200 goes out. The portal's "Test
        # Webhook" button sends a synthetic TEST event; it takes the same path
        # and submits a neutral prediction (accepted by the API, never scored)
        # so the test exercises your full receive -> submit loop.
        await predict_and_submit.spawn.aio(event, webhook_id)
        return Response(status_code=200)

    return api
