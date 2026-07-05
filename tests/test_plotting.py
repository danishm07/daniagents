"""The event-calendar chart. Uses a non-interactive backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from examples.frames import events_frame
from examples.plotting import plot_event_calendar
from examples.schemas import CalendarEvent


def test_plot_returns_fig_and_labels_every_type(sample_events: list[dict]) -> None:
    df = events_frame([CalendarEvent.model_validate(e) for e in sample_events])
    fig, ax = plot_event_calendar(df)
    try:
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        # Every event type present in the data is represented in the legend, so
        # identity never rides on color alone.
        assert set(df["event_type"]) <= legend_labels
    finally:
        matplotlib.pyplot.close(fig)


def test_plot_handles_empty_frame() -> None:
    fig, ax = plot_event_calendar(events_frame([]))
    try:
        assert ax.get_legend() is None  # nothing to plot, no legend
    finally:
        matplotlib.pyplot.close(fig)
