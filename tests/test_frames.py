"""Display-frame helpers."""

from __future__ import annotations

from examples.frames import events_frame, health_frame, manifest_frame
from examples.schemas import ArchiveManifest, CalendarEvent, SubmissionHealth


def test_events_frame_sorts_and_flattens(sample_events: list[dict]) -> None:
    # event_type is an open string set: an as-yet-unknown type (here with no
    # focal assets) must flow through without any special-casing.
    future_event = {
        "event_id": "0f8c2a10-1b2c-4d5e-8a9b-0000000000ff",
        "event_type": "SOME_FUTURE_TYPE",
        "timing_category": "SCHEDULED",
        "event_datetime": "2026-01-28T13:30:00Z",
        "focal_assets": [],
    }
    events = [CalendarEvent.model_validate(e) for e in [*sample_events, future_event]]
    df = events_frame(events)

    assert list(df.columns) == [
        "event_datetime",
        "knowledge_cutoff",
        "event_type",
        "timing_category",
        "tickers",
        "n_assets",
        "event_id",
    ]
    # Sorted ascending by time.
    assert df["event_datetime"].is_monotonic_increasing
    # Focal assets flattened to a ticker string + count.
    earnings = df[df["event_type"] == "EARNINGS_RELEASE"].iloc[0]
    assert earnings["n_assets"] == 1
    assert earnings["tickers"] != ""
    # An event with no focal assets renders as empty tickers, zero count.
    no_assets = df[df["event_type"] == "SOME_FUTURE_TYPE"].iloc[0]
    assert no_assets["tickers"] == ""
    assert no_assets["n_assets"] == 0


def test_events_frame_empty() -> None:
    df = events_frame([])
    assert df.empty
    assert list(df.columns)[:2] == ["event_datetime", "knowledge_cutoff"]


def test_manifest_frame_hides_urls_and_sorts() -> None:
    manifest = ArchiveManifest.model_validate(
        {
            "files": [
                {
                    "event_type": "SOME_FUTURE_TYPE",
                    "quarter": "2026Q2",
                    "key": "k2",
                    "url": "https://cdn/secret2",
                    "events": 3,
                    "bytes": 2_000_000,
                },
                {
                    "event_type": "EARNINGS_RELEASE",
                    "quarter": "2025Q3",
                    "key": "k1",
                    "url": "https://cdn/secret1",
                    "events": 5,
                    "bytes": 863,
                },
            ]
        }
    )
    df = manifest_frame(manifest)
    # Signed URLs never surface in the display frame.
    assert not any("url" in c for c in df.columns)
    # Sorted by (event_type, quarter): EARNINGS_RELEASE before SOME_FUTURE_TYPE.
    assert df.iloc[0]["event_type"] == "EARNINGS_RELEASE"
    assert df.iloc[0]["size_mb"] == 0.0  # 863 bytes rounds to 0.00 MB
    assert df.iloc[1]["size_mb"] == 2.0


def test_health_frame_groups_and_drops_nulls() -> None:
    health = SubmissionHealth.model_validate(
        {"submission_id": "s1", "webhook_n_2xx": 10, "submission_n_total": 4}
    )
    df = health_frame(health)
    # submission_id is dropped; nulls are dropped; groups are labeled.
    assert "submission_id" not in df["metric"].to_numpy()
    assert set(df["group"]) == {"webhook", "submission"}
    assert df.loc[df["metric"] == "webhook_n_2xx", "value"].iloc[0] == 10
