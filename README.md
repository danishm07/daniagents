# Explaining Markets — examples

[![CI](https://github.com/explaining-markets/examples/actions/workflows/ci.yml/badge.svg)](https://github.com/explaining-markets/examples/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Notebook examples for exploring the [Explaining Markets](https://explainingmarkets.ai/)
competition API and its historical data archive. Notebook-first, with the repeated
API / parsing / plotting boilerplate factored into a small `examples` package so the
notebooks stay short and readable.

## What's here

| Notebook | What it covers |
|---|---|
| [`notebooks/00_api_quickstart.ipynb`](notebooks/00_api_quickstart.ipynb) | The read endpoints — events calendar (table + chart), submission health, archive manifest, webhook self-test |
| [`notebooks/01_historical_archive.ipynb`](notebooks/01_historical_archive.ipynb) | Download, cache, and load the historical archive, then reproduce the baseline scoring regressions (Koijen & Levy WP, Table 3) with the competition's exact scoring transform |

Both run top-to-bottom. With an API key they hit the live API; **without one they
fall back to a small bundled sample** (`data/sample/`), so you can run them — and CI
can execute them — with no credentials.

> **Submitting predictions is out of scope here.** `POST /predictions` places a
> real, scored entry and belongs in a *deployed* webhook handler. Start from the
> [`starter-modal`](https://github.com/explaining-markets/starter-modal) (or
> `starter-railway`) repos for that.

## Setup

This repo uses [uv](https://docs.astral.sh/uv/). From a clone:

```bash
uv sync
cp .env.example .env      # then paste your submission's API key (optional)
uv run jupyter lab
```

No key yet? Skip the `.env` step and everything still runs in sample mode.

## Using the helpers outside a notebook

```python
from examples import Client
from examples.frames import events_frame

with Client.from_env() as client:
    events = client.events()

events_frame(events).head()
```

## Repo layout

```text
notebooks/   the numbered, narrative examples
src/examples/  helper package: config · client · schemas · frames · plotting · archive · scoring
scripts/     small runnable scripts (e.g. headless archive download)
tests/       offline unit tests (mocked HTTP + bundled sample data)
data/
  sample/    tiny, committed, illustrative data
  archive/   gitignored download cache for real archive files
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # type-check
uv run pytest                # offline unit tests
uv run pytest --nbmake notebooks/   # execute the notebooks (sample mode)
uv run pre-commit install    # strip notebook outputs + lint on commit
```

CI runs all of the above on every push and pull request. Live tests (`-m live`)
need `EM_API_KEY` and are excluded by default.

## Related repositories

- [`starter-modal`](https://github.com/explaining-markets/starter-modal) — deploy a webhook handler on Modal
- [`starter-railway`](https://github.com/explaining-markets/starter-railway) — the same, on other platforms
- [`Official baselines`](https://github.com/explaining-markets/baseline-earnings-summary) — reference agents to measure yourself against
