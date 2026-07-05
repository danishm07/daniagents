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
