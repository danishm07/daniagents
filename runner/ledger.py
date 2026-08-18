"""The single accounting path for metered spend. Provider API is ground truth.

**Standing correction, 2026-08-18.** Every spend report in this project before
this module existed was wrong, by a factor of **3.2×**. The old accounting
reconstructed spend by summing ``prompt_tokens``/``completion_tokens`` out of
the JSONL columns in ``data/{arms,reads,frontier}`` and reported **$10.83**.
OpenRouter's own API reported **$37.22 of $37.00 consumed** — the account was
exhausted, and a run was authorised against a budget that did not exist.

The failure is structural, not arithmetic. File reconstruction can only see
spend that happened to leave a token column behind. It cannot see DSPy replay
runs, aborted sweeps that wrote nothing, baseline replays under a different
writer, or any new script — like the ACE pilot — that had not been taught to
write one. Each of those is invisible *by construction*, so the reconstruction
is a **lower bound** and was being read as a total.

Two rules follow, and they are the whole point of this module:

1. **Budget decisions reconcile against the provider, never against files.**
   :func:`remaining` asks OpenRouter. :func:`reconstructed` is retained only as
   a lower bound and as the input to :func:`reconcile`, whose job is to make the
   gap visible rather than to be believed.
2. **Every metered call routes through one path.** :func:`record` is that path.
   A caller that does not use it is invisible again, which is how this happened.

    from runner import ledger
    ledger.guard(estimated_usd=2.70)      # refuses if the provider says no
    ...
    ledger.record("google/gemini-2.5-flash-lite", 5984, 758, source="ace.generator")
    ledger.report()
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = ROOT / "runs" / "ledger.jsonl"

#: Directories whose JSONL columns carry token counts. Retained for the
#: lower-bound reconstruction only — see the module docstring on why this is not
#: a total.
SPEND_DIRS = ("arms", "reads", "frontier")

_lock = threading.Lock()


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def provider_balance(timeout: float = 20.0) -> dict:
    """What this API key can still spend. The number budget decisions are made on.

    **Read the endpoint choice before changing it — the obvious one is wrong.**

    ``/api/v1/credits`` returns ``total_credits`` and ``total_usage``. Those are
    **account-lifetime** figures: credits ever purchased, and spend ever made
    across every key on the account. They are *not* a balance. On 2026-08-18 they
    read 37.00 and 37.22, which was taken to mean "the account is exhausted and
    overdrawn" and used to declare a funded run unaffordable. It meant nothing of
    the sort — an account that has, over its lifetime, spent about what it bought
    is the normal steady state, and the near-equality of the two numbers should
    have been read as a units warning rather than as a balance.

    ``/api/v1/key`` returns this key's own ``usage``, its ``limit`` (a per-key
    spending cap) and ``limit_remaining``. That last figure is what actually
    gates a call, and it is what produced the ``402`` errors: the key was capped
    at $10 with $9.55 spent, while the account itself was fine.

    So: **key limit is the binding constraint; account credits are context.**
    Both are reported, and ``remaining`` is the key's.

    Returns ``{"ok": False, ...}`` rather than raising, because a balance check
    that fails must not be silently read as "plenty left" by a caller that
    swallowed the exception.
    """
    import httpx
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"ok": False, "error": "no OPENROUTER_API_KEY"}
    headers = {"Authorization": f"Bearer {key}"}

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key", headers=headers, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()["data"]
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    used = float(data.get("usage") or 0.0)
    limit = data.get("limit")
    limit_remaining = data.get("limit_remaining")

    out = {
        "ok": True,
        "key_usage": used,
        "key_usage_daily": float(data.get("usage_daily") or 0.0),
        "key_limit": None if limit is None else float(limit),
        # An uncapped key reports limit=None; then the account's credits are the
        # only ceiling and "remaining" is unbounded from the key's perspective.
        "remaining": (
            float(limit_remaining) if limit_remaining is not None else float("inf")
        ),
        "uncapped": limit is None,
    }

    # Account-lifetime context. Explicitly NOT a balance — see the docstring.
    try:
        credits = httpx.get(
            "https://openrouter.ai/api/v1/credits", headers=headers, timeout=timeout
        ).json()["data"]
        out["account_lifetime_purchased"] = float(credits.get("total_credits", 0.0))
        out["account_lifetime_used"] = float(credits.get("total_usage", 0.0))
    except Exception:
        pass
    return out


def remaining() -> float:
    """USD available, from the provider. ``0.0`` if the balance cannot be read.

    Failing closed is deliberate: the alternative failure mode is the one that
    just happened.
    """
    balance = provider_balance()
    return max(balance["remaining"], 0.0) if balance["ok"] else 0.0


# --------------------------------------------------------------------------
# The one path
# --------------------------------------------------------------------------


def record(model: str, prompt_tokens: int, completion_tokens: int, *,
           source: str, note: str = "") -> float:
    """Log one metered call and return its cost. **Every caller uses this.**

    Priced from the local table for an immediate per-call figure; the provider
    remains authoritative for the total, and :func:`reconcile` reports the gap
    between the two rather than assuming they agree.
    """
    import frontier

    price_in, price_out = frontier.PRICES.get(model, (1.0, 5.0))
    usd = (prompt_tokens or 0) * price_in / 1e6 + (completion_tokens or 0) * price_out / 1e6
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LOG.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "usd": usd,
                    "source": source,
                    "note": note,
                }
            )
            + "\n"
        )
    return usd


def logged() -> dict:
    """What this ledger has seen since it started, by source."""
    if not LOG.exists():
        return {"total_usd": 0.0, "calls": 0, "by_source": {}}
    total, calls, by_source = 0.0, 0, {}
    for line in LOG.open():
        if not line.strip():
            continue
        row = json.loads(line)
        total += row.get("usd", 0.0)
        calls += 1
        key = row.get("source", "unknown")
        entry = by_source.setdefault(key, {"usd": 0.0, "calls": 0})
        entry["usd"] += row.get("usd", 0.0)
        entry["calls"] += 1
    return {"total_usd": total, "calls": calls, "by_source": by_source}


# --------------------------------------------------------------------------
# The lower bound, kept honest about being one
# --------------------------------------------------------------------------


def reconstructed() -> float:
    """Spend inferable from token columns on disk. **A lower bound, not a total.**

    Blind to anything that did not leave a column: DSPy runs, aborted sweeps,
    replays under another writer, and any script not yet taught to write one.
    """
    import frontier

    total = 0.0
    for directory in SPEND_DIRS:
        for path in (ROOT / "data" / directory).glob("*.jsonl"):
            stem = path.stem.split("__")[-1]
            model = stem.replace("_", "/", 1) if "_" in stem else stem
            price_in, price_out = frontier.PRICES.get(model, (0.10, 0.40))
            tin = tout = 0
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                tin += row.get("prompt_tokens") or 0
                tout += row.get("completion_tokens") or 0
            total += tin * price_in / 1e6 + tout * price_out / 1e6
    return total


def reconcile() -> dict:
    """Provider vs reconstruction vs this ledger. The gap is the point."""
    balance = provider_balance()
    files = reconstructed()
    mine = logged()
    out = {
        "provider": balance,
        "reconstructed_lower_bound_usd": files,
        "this_ledger_usd": mine["total_usd"],
        "this_ledger_calls": mine["calls"],
        "by_source": mine["by_source"],
    }
    if balance["ok"]:
        out["unattributed_usd"] = balance["key_usage"] - files
        out["reconstruction_ratio"] = (
            (balance["key_usage"] / files) if files else float("inf")
        )
    return out


def guard(estimated_usd: float, *, margin: float = 1.25) -> None:
    """Refuse to start a run the provider cannot fund.

    ``margin`` because an estimate that is exactly right has never happened
    here — today's run was estimated at $2.42 against a believed $13.19 and the
    account was already empty.
    """
    balance = provider_balance()
    if not balance["ok"]:
        raise RuntimeError(
            f"cannot read the provider balance ({balance.get('error')}) — refusing to spend. "
            "Failing closed is deliberate; the alternative is what caused the 3.2x error."
        )
    needed = estimated_usd * margin
    if needed > balance["remaining"]:
        raise RuntimeError(
            f"estimated ${estimated_usd:.2f} (x{margin:g} margin = ${needed:.2f}) exceeds "
            f"${balance['remaining']:.2f} spendable on this key "
            f"(${balance['key_usage']:.2f} used of a ${balance['key_limit']:.2f} per-key cap). "
            "Raise the key's limit in the OpenRouter dashboard, or use an uncapped key — "
            "this is a KEY cap, not an empty account."
        )


def report() -> dict:
    data = reconcile()
    balance = data["provider"]
    print("=" * 78)
    print("SPEND LEDGER")
    print("=" * 78)
    if balance["ok"]:
        cap = "uncapped" if balance["uncapped"] else f"${balance['key_limit']:.2f} cap"
        left = "unlimited" if balance["uncapped"] else f"${balance['remaining']:.2f}"
        print(f"  KEY (binding constraint)              "
              f"${balance['key_usage']:.2f} used of {cap}  -> {left} spendable")
        print(f"    today                               ${balance['key_usage_daily']:.2f}")
        if "account_lifetime_used" in balance:
            print(f"  account lifetime (context, NOT a balance) "
                  f"${balance['account_lifetime_used']:.2f} used / "
                  f"${balance['account_lifetime_purchased']:.2f} ever purchased")
    else:
        print(f"  provider                              UNREADABLE: {balance.get('error')}")
    print(f"  reconstructed from files (LOWER BOUND) ${data['reconstructed_lower_bound_usd']:.2f}")
    print(f"  this ledger ({data['this_ledger_calls']} calls)"
          f"{'':>{max(0, 24 - len(str(data['this_ledger_calls'])))}}${data['this_ledger_usd']:.2f}")
    if "unattributed_usd" in data:
        print(f"  UNATTRIBUTED                          ${data['unattributed_usd']:.2f}  "
              f"(reconstruction understates by {data['reconstruction_ratio']:.1f}x)")
    if data["by_source"]:
        print("\n  by source:")
        for source, entry in sorted(data["by_source"].items(), key=lambda kv: -kv[1]["usd"]):
            print(f"    {source:<28} ${entry['usd']:>7.4f}  {entry['calls']} calls")
    return data


if __name__ == "__main__":
    report()
