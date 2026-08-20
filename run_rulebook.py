"""Third place's design: one frontier call plus a frozen curated rulebook."""
import json, sys, threading, time, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness, arms as A, reads, predict, champion
from runner import ledger

MODEL = "anthropic/claude-haiku-4.5"
RB = pathlib.Path("Rulebook.md").read_text()
ids = json.load(open("/tmp/replicate_ids.json"))          # first 2,000 confirmation events
by = {e["event_id"]: e for q in harness.DEV_QUARTERS for e in harness.events_for(q)}
events = A.tag_quarters([by[i] for i in ids if i in by])

def run(tag, ctx, workers=14):
    path = pathlib.Path(f"data/ace/rb__{tag}.jsonl")
    done = {json.loads(l)["event_id"] for l in path.open()} if path.exists() else set()
    todo = [e for e in events if e["event_id"] not in done]
    print(f"[{tag}] cached {len(done)}, to run {len(todo)}", flush=True)
    if not todo: return
    client = reads._client(); lock = threading.Lock()
    t0, cost, tin, tout, fails = time.time(), 0.0, 0, 0, 0
    def one(e):
        prompt = A.build_prompt(e, ctx)
        if prompt is None:   # no facts: production submits 0.5 without calling
            return {"event_id": e["event_id"], "prediction": 0.5,
                    "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        champion._throttle.wait(champion._throttle.estimate(2600))
        r = client.chat.completions.parse(model=MODEL, messages=[
            {"role":"system","content":predict.SYSTEM_PROMPT},
            {"role":"user","content":prompt}],
            response_format=reads.Direct, max_tokens=900)
        u = r.usage; p = r.choices[0].message.parsed
        return {"event_id": e["event_id"],
                "prediction": float("nan") if p is None else p.predicted_percentile,
                "prompt_tokens": u.prompt_tokens if u else 0,
                "completion_tokens": u.completion_tokens if u else 0,
                "cost": float(getattr(u,"cost",0) or 0) if u else 0.0}
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        for i, fut in enumerate(as_completed([pool.submit(one,e) for e in todo]), 1):
            try: row = fut.result()
            except Exception as ex:
                fails += 1
                if fails <= 3: print(f"  [fail] {type(ex).__name__}: {str(ex)[:90]}", flush=True)
                continue
            with lock: fh.write(json.dumps(row)+"\n"); fh.flush()
            tin += row["prompt_tokens"]; tout += row["completion_tokens"]; cost += row["cost"]
            if i % 250 == 0:
                r_ = i/(time.time()-t0)
                print(f"  {i}/{len(todo)} {r_:.1f}/s eta {(len(todo)-i)/r_/60:.1f}m ${cost:.2f}", flush=True)
    usd = ledger.record(MODEL, tin, tout, source=f"rulebook.{tag}", usd=cost or None)
    print(f"[{tag}] done, {fails} failures, ${usd:.4f}", flush=True)

champion._throttle = champion._Throttle(600, 6_000_000)
ledger.guard(estimated_usd=8.0, margin=1.0)
run("A_norule", "")
run("B_rulebook", RB)
