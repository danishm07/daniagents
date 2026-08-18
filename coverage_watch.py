"""Coverage monitor. Free, and the only way to tell a gap from an outage.

Modal's log retention is minutes, the leaderboard recomputes daily, and a missed
event is permanent and mean-imputed on the board that decides prizes. So the
question "are we receiving everything?" needs a persistent local record polled
over time, not a log tail.

Writes one row per poll to data/coverage/watch.jsonl: our prediction_log count,
the leaderboard's count for us and for a full-coverage peer, and the calendar
size. The delta between polls is the delivery rate; a peer's delta is what it
should be.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]
from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

OUT = Path(__file__).parent / "data" / "coverage" / "watch.jsonl"
BOARD = "https://df48ooei1lq8t.cloudfront.net/leaderboards/v1/lb_global/2026Q3-CONTEST.json"


def poll() -> dict:
    import modal

    from explaining_markets.config import Config

    log = modal.Dict.from_name("em-prediction-log", create_if_missing=True)
    rows = [v for _, v in log.items() if not str(v.get("event_id", "")).startswith("diag")]

    cfg = Config.from_env()
    calendar = httpx.get(f"{cfg.api_base_url}/events",
                         headers={"X-API-Key": cfg.api_key}, timeout=20).json()

    board = httpx.get(BOARD, timeout=20).json()
    ours = next((r for r in board["rows"] if r["public_name"] == "danigents"), {})
    top = max(board["rows"], key=lambda r: r["n_obs"])

    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "prediction_log_events": len(rows),
        "rungs": {r: sum(1 for x in rows if x.get("rung") == r)
                  for r in {x.get("rung") for x in rows}},
        "neutral_rate": (sum(1 for x in rows if x.get("rung") == "neutral") / len(rows)) if rows else None,
        "unsubmitted": sum(1 for x in rows if not x.get("submitted")),
        "board_computed_at": board["computed_at"],
        "board_scorable": board["scorable_events"],
        "our_n_obs": ours.get("n_obs"),
        "best_peer_n_obs": top["n_obs"],
        "calendar_upcoming": len(calendar),
    }


if __name__ == "__main__":
    row = poll()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    history = [json.loads(l) for l in OUT.open() if l.strip()]
    if len(history) > 1:
        prev = history[-2]
        print(f"\nsince last poll ({prev['at']}): "
              f"+{row['prediction_log_events'] - prev['prediction_log_events']} events processed, "
              f"peer +{(row['best_peer_n_obs'] or 0) - (prev['best_peer_n_obs'] or 0)}")
