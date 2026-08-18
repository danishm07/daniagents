"""SEC EDGAR (data.sec.gov) retrieval + features — Agent B.

Everything external goes through :func:`sources.fetch` so the §07/§10 audit log
records it. Raw JSON is cached gzipped under ``data/edgar/`` so re-runs are free
and the rate limiter is only paid once.

The one field that matters for cutoff filtering is **``acceptanceDateTime``**.
``filingDate`` is a legal construct: anything accepted after 17:30 ET rolls to
the next business day, so it disagrees with wall-clock reality on a large
minority of filings. Every in-bounds test in this module uses
``acceptanceDateTime`` (which EDGAR serves in ET, no offset, as
``YYYY-MM-DDTHH:MM:SS.000Z`` — the trailing Z is a lie, see :func:`_accept_ts`).
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import sources

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "edgar"
UA = "explaining-markets research danishtaher7@gmail.com"

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

_rate_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.11  # 10 req/s ceiling, with headroom


def _throttle() -> None:
    with _rate_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _cache_path(key: str) -> Path:
    return CACHE / f"{key}.json.gz"


def cached(key: str) -> Any | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _store(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(payload, fh)
    tmp.replace(path)


def get_json(url: str, key: str, *, notes: str = "", tries: int = 3) -> Any | None:
    """Cached, throttled, audited GET of a JSON endpoint. ``None`` on 404.

    Logged ``point_in_time=False``: the endpoint serves the *current* full
    history of a filer, not a snapshot as of any cutoff. Point-in-time is
    restored downstream by filtering on ``acceptanceDateTime``, and every
    per-event slice is logged separately with its own window.
    """
    hit = cached(key)
    if hit is not None:
        return hit if hit != {"__404__": True} else None

    def loader() -> Any:
        last: Exception | None = None
        for attempt in range(tries):
            _throttle()
            try:
                r = _session.get(url, timeout=30)
            except Exception as exc:  # network flake
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 404:
                return {"__404__": True}
            if r.status_code in (429, 403, 500, 502, 503):
                time.sleep(2.0 * (attempt + 1))
                last = RuntimeError(f"{r.status_code} {url}")
                continue
            r.raise_for_status()
            return r.json()
        raise last or RuntimeError(f"failed {url}")

    payload = sources.fetch(
        source="sec-edgar",
        endpoint=url.replace("https://", ""),
        loader=loader,
        point_in_time=False,
        notes=notes
        or "full filer history; cutoff enforced downstream on acceptanceDateTime",
    )
    _store(key, payload)
    return None if payload == {"__404__": True} else payload


# ---------------------------------------------------------------------------
# ticker -> CIK -> SIC
# ---------------------------------------------------------------------------

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def ticker_map() -> pd.DataFrame:
    """``ticker, cik, title, exchange`` for every SEC filer with a ticker."""
    rows = []
    exch = get_json(TICKERS_EXCHANGE_URL, "ref/company_tickers_exchange",
                    notes="reference: ticker->CIK->exchange (current values)")
    if exch:
        fields = exch["fields"]
        idx = {f: i for i, f in enumerate(fields)}
        for row in exch["data"]:
            rows.append({
                "ticker": str(row[idx["ticker"]]).upper(),
                "cik": int(row[idx["cik"]]),
                "title": row[idx["name"]],
                "exchange": row[idx["exchange"]],
            })
    plain = get_json(TICKERS_URL, "ref/company_tickers",
                     notes="reference: ticker->CIK (current values)")
    known = {r["ticker"] for r in rows}
    if plain:
        for rec in plain.values():
            t = str(rec["ticker"]).upper()
            if t not in known:
                rows.append({"ticker": t, "cik": int(rec["cik_str"]),
                             "title": rec["title"], "exchange": None})
    return pd.DataFrame(rows).drop_duplicates(subset="ticker")


BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"


def browse_company(ticker: str) -> dict | None:
    """Fallback ticker lookup for filers absent from ``company_tickers.json``.

    Measured: 72 of the archive's 2,518 tickers (AEP, BK, EXAS, IAC, ...) are
    missing from that file. The legacy browse-edgar atom endpoint resolves them
    and carries ``assigned-sic`` inline.
    """
    key = f"browse/{ticker.replace('/', '_')}"
    hit = cached(key)
    if hit is not None:
        return None if hit == {"__404__": True} else hit

    def loader() -> Any:
        import re

        for attempt in range(4):
            _throttle()
            try:
                r = _session.get(
                    BROWSE,
                    params={"action": "getcompany", "ticker": ticker, "type": "10-K",
                            "dateb": "", "owner": "include", "count": "1",
                            "output": "atom"},
                    timeout=20,
                )
            except Exception:
                time.sleep(1.0 * (attempt + 1))
                continue
            text = r.text
            out: dict[str, Any] = {}
            for tag in ("cik", "assigned-sic", "assigned-sic-desc", "conformed-name"):
                m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
                if m:
                    out[tag] = m.group(1).strip()
            return out or {"__404__": True}
        return {"__404__": True}

    payload = sources.fetch(
        source="sec-edgar", endpoint="cgi-bin/browse-edgar", loader=loader,
        point_in_time=False,
        notes=f"reference ticker->CIK/SIC fallback for {ticker} (current values)",
    )
    _store(key, payload)
    return None if payload == {"__404__": True} else payload


def submissions(cik: int) -> dict | None:
    """``data.sec.gov/submissions/CIK##########.json`` — filing index + SIC."""
    return get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions/CIK{cik:010d}",
        notes="filer submission index (filings.recent) + SIC; "
              "cutoff enforced downstream on acceptanceDateTime",
    )


def companyfacts(cik: int) -> dict | None:
    """``data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` — all XBRL facts."""
    return get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        f"companyfacts/CIK{cik:010d}",
        notes="all reported XBRL facts; cutoff enforced downstream via "
              "accn -> acceptanceDateTime join against submissions",
    )


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------


def _accept_ts(value: str) -> pd.Timestamp:
    """EDGAR ``acceptanceDateTime`` -> UTC. The ``Z`` is honest: the clock is UTC.

    Measured, and worth stating because getting it wrong shifts every filing by
    four hours. On 78,744 8-K/10-Q/10-K/DEF 14A filings from 400 cached filers,
    apply EDGAR's own rule ("accepted after 17:30 ET gets the next business
    day's ``filingDate``") as a discriminator:

        raw interpreted as UTC  -> predicts the observed roll on **99.72%**
        raw interpreted as ET   -> predicts it on 43.88%

    Converted to ET under the UTC reading, the acceptance-hour histogram peaks
    hard at 16:00 ET (27,349 of 78,744) — the post-close 8-K rush — and is
    essentially empty 22:00-05:00 ET. Under the ET reading the modal hour is
    20:00, which is not when anyone files.
    """
    s = str(value).replace("Z", "")
    return pd.Timestamp(s).tz_localize("UTC")


def filings_frame(cik: int) -> pd.DataFrame:
    """Flatten ``filings.recent`` into a frame with a UTC ``accepted`` column."""
    sub = submissions(cik)
    if not sub:
        return pd.DataFrame()
    rec = sub.get("filings", {}).get("recent", {})
    if not rec or not rec.get("accessionNumber"):
        return pd.DataFrame()
    keep = ["accessionNumber", "filingDate", "acceptanceDateTime", "form",
            "primaryDocument", "items", "reportDate", "size"]
    frame = pd.DataFrame({k: rec[k] for k in keep if k in rec})
    frame["accepted"] = [
        _accept_ts(v) if v else pd.NaT for v in frame["acceptanceDateTime"]
    ]
    frame["cik"] = cik
    return frame
