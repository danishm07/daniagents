# Data

This repo does **not** ship the competition's historical dataset. The real data
lives behind the API — this folder holds only a tiny sample for offline reading
and a cache directory for what you download.

## `sample/` — tiny, committed, illustrative

Just enough data for the notebooks to run top-to-bottom **without an API key**
(this is also what CI executes):

- `events_sample.json` — a handful of calendar events, exactly the shape returned
  by `GET /events`.
- `archive_EARNINGS_RELEASE_2025Q3.jsonl.gz` — five illustrative historical
  earnings events in the archive's gzip-JSONL format. The record shape mirrors the
  documented event payload with an inlined disclosure (`facts`). It is a
  **stand-in for exposition** — the real archive is served by the API and its
  lines may include additional fields. Treat the API as the source of truth.
- `transcript_sample.json` — a **synthetic** earnings-call transcript for a
  fictional company (Northwind Logistics). Real transcripts are licensed and are
  not distributed with this repo; this one exists so
  `notebooks/02_earnings_call_facts.ipynb` and `examples.summary.format_transcript`
  have something to run on offline. The shape mirrors the raw record the
  production summarizer renders.
- `summary_sample.json` — a **synthetic** example of the artifact the summarizer
  publishes per event: `event_id`, `response.facts` (the ten sentences that become
  the disclosure item), `response.parse_note`, and `metadata`. The facts were
  written by hand from `transcript_sample.json`; they are not model output.

Keep anything here small. Never commit bulk data.

## `archive/` — downloaded, gitignored

Where `download_archive(...)` and `notebooks/01_historical_archive.ipynb` cache
the real gzip-JSONL files pulled from `GET /archive`. These can be large, so the
folder's contents are gitignored (the folder itself is kept via `.gitkeep`).

Get the data from the API, not from git:

```python
from examples import Client
from examples.archive import download_archive, load_archive

with Client.from_env() as client:
    manifest = client.archive_manifest()
    download_archive(manifest, "data/archive", client=client)

df = load_archive("data/archive")
```

See the [API reference](https://api-beta.explainingmarkets.ai/) and the repo
README for how the archive endpoints work.
