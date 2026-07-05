"""Plotting helpers for the notebooks.

One chart, done carefully: a stacked daily bar of upcoming events by type. The
job is *counts over time, split by category*, so the form is a time-bucketed
stacked bar. Colors are the Okabe-Ito colorblind-safe set, assigned to event
types in a **fixed, stable order** (never by rank), with a 9th-and-beyond type
folding into a neutral "Other" — so identity never rides on color alone and the
palette never cycles.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Okabe-Ito, ordered strongest-contrast-first. Reserve a neutral gray for "Other".
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]
_OTHER_COLOR = "#9AA0A6"
_OTHER_LABEL = "Other"

_GRID = "#D7DBE0"
_INK = "#3C4043"


def plot_event_calendar(
    events_df: pd.DataFrame,
    *,
    title: str = "Upcoming events by day",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Stacked daily bar chart of events by type from an :func:`events_frame`.

    Expects the columns produced by :func:`examples.frames.events_frame`
    (``event_datetime``, ``event_type``). Returns ``(figure, axes)`` so the
    caller can tweak or save.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = cast(Figure, ax.figure)

    if events_df.empty:
        ax.text(0.5, 0.5, "No events in range", ha="center", va="center", color=_INK)
        ax.axis("off")
        return fig, ax

    counts = _daily_counts_by_type(events_df)
    colors = _color_map(list(counts.columns))

    bottom = pd.Series(0, index=counts.index, dtype=float)
    for event_type in counts.columns:
        ax.bar(
            counts.index,
            counts[event_type],
            bottom=bottom,
            width=0.8,
            color=colors[event_type],
            edgecolor="white",
            linewidth=0.5,  # 2px-ish surface gap between stacked segments
            label=event_type,
        )
        bottom += counts[event_type]

    _style(ax, title, ymax=int(bottom.max()))
    # Rotate date labels so they never collide, even on a narrow figure.
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title="Event type",
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig, ax


def _daily_counts_by_type(events_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot events into a day-by-event_type count matrix, folding rare types into 'Other'."""
    df = events_df.copy()
    df["day"] = pd.to_datetime(df["event_datetime"], utc=True).dt.date
    df["bucket"] = _fold_rare_types(df["event_type"])
    wide = df.groupby(["day", "bucket"], observed=True).size().unstack(fill_value=0).sort_index()
    # Stable column order: fixed categories first (by total), 'Other' always last.
    ordered = [c for c in wide.sum().sort_values(ascending=False).index if c != _OTHER_LABEL]
    if _OTHER_LABEL in wide.columns:
        ordered.append(_OTHER_LABEL)
    return wide[ordered]


def _fold_rare_types(event_type: pd.Series) -> pd.Series:
    """Keep the palette-many most common types; everything else becomes 'Other'."""
    keep = event_type.value_counts().index[: len(_PALETTE)]
    return event_type.where(event_type.isin(keep), _OTHER_LABEL)


def _color_map(event_types: list[str]) -> dict[str, str]:
    """Assign palette colors in the given (already stable) order; 'Other' is gray."""
    mapping: dict[str, str] = {}
    palette_i = 0
    for et in event_types:
        if et == _OTHER_LABEL:
            mapping[et] = _OTHER_COLOR
        else:
            mapping[et] = _PALETTE[palette_i % len(_PALETTE)]
            palette_i += 1
    return mapping


def _style(ax: Axes, title: str, *, ymax: int) -> None:
    ax.set_title(title, fontsize=12, color=_INK, loc="left", pad=12)
    ax.set_ylabel("Events", fontsize=9, color=_INK)
    ax.margins(x=0.01)
    ax.tick_params(colors=_INK, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Integer y ticks — counts are whole numbers.
    if ymax > 0:
        step = max(1, ymax // 5)
        ax.set_yticks(range(0, ymax + step, step))
