"""Display-frame helpers."""

from __future__ import annotations

from examples.frames import events_frame, health_frame, manifest_frame
from examples.schemas import ArchiveManifest, CalendarEvent, SubmissionHealth


def test_events_frame_sorts_and_flattens(sample_events: list[dict]) -> None:
    events = [CalendarEvent.model_validate(e) for e in sample_events]
    df = events_frame(events)

    assert list(df.columns) == [
        "event_datetime",
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
    # A macro event with no focal assets renders as empty tickers, zero count.
    macro = df[df["event_type"] == "MACRO_DATA_CPI"].iloc[0]
    assert macro["tickers"] == ""
    assert macro["n_assets"] == 0


def test_events_frame_empty() -> None:
    df = events_frame([])
    assert df.empty
    assert list(df.columns)[:2] == ["event_datetime", "event_type"]


def test_manifest_frame_hides_urls_and_sorts() -> None:
    manifest = ArchiveManifest.model_validate(
        {
            "files": [
                {
                    "event_type": "FOMC_MEETING",
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
    # Sorted by (event_type, quarter): EARNINGS before FOMC.
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
