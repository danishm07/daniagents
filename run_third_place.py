"""Third place's published recipe, at confirmation-partition scale.

Their model card lists three things together: the focal ticker's realized
abnormal returns at its own prior earnings events, the factual summary published
for its most recent previous event, and the distribution of realized abnormal
returns across the previous completed quarter. `arms.ctx_third_place`
concatenates exactly those three, and `arms.build_prompt` prefixes them to the
deployed prompt — so this is the full recipe on the production path, not a
component of it.

Scored three ways because 2025Q4 has no prior quarter and therefore no context
at all, which makes it a free placebo rather than a third of a diluted sample.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms as A, harness, champion
from runner import ledger, schedule as S

conf = {e["event_id"] for v in S.confirmation_events().values() for e in v}
done = {json.loads(l)["event_id"] for l in open("data/arms/third_place__google_gemini-2.5-flash-lite.jsonl") if l.strip()}
events = [e for e in A.tag_quarters([e for q in harness.DEV_QUARTERS for e in harness.events_for(q)])
          if e["event_id"] in conf and e["event_id"] not in done]
print(f"third_place on the confirmation partition: {len(events)} new events")
ledger.guard(estimated_usd=0.55)
champion._throttle = champion._Throttle(500, 4_000_000)
before = A.cost("third_place", "google/gemini-2.5-flash-lite")
A.run_arm("third_place", "google/gemini-2.5-flash-lite", events, workers=14)
after = A.cost("third_place", "google/gemini-2.5-flash-lite")
usd = ledger.record("google/gemini-2.5-flash-lite", after[0]-before[0], after[1]-before[1],
                    source="third_place.confirmation",
                    note=f"full third-place recipe, {len(events)} confirmation events")
print(f"\nledger: ${usd:.4f}")
