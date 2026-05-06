#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =============================================================================
# KONFIG
# =============================================================================
DEMAND_BASE_DIR = Path("/home/endata/PycharmProjects/pypsa-ee/resources")
DEMAND_DIR_PATTERN = "demand_{year}"              # z.B. demand_2027
DEMAND_FILENAME = "electricity_demand_non_hist.csv"


# =============================================================================
# HELPERS
# =============================================================================
_YEAR_RE = re.compile(r"^demand_(\d{4})$")


def find_available_years(base_dir: Path) -> List[int]:
    years: List[int] = []
    if not base_dir.exists():
        return years
    for p in base_dir.iterdir():
        if p.is_dir():
            m = _YEAR_RE.match(p.name)
            if m:
                years.append(int(m.group(1)))
    return sorted(years)


def demand_file_for_year(base_dir: Path, year: int) -> Path:
    folder = base_dir / DEMAND_DIR_PATTERN.format(year=year)
    return folder / DEMAND_FILENAME


@st.cache_data(show_spinner=False)
def load_demand_csv(csv_path: str) -> pd.DataFrame:
    """
    Robust loader:
    - akzeptiert 'Date' oder erste Spalte als Zeitstempel
    - parst Datetime, sortiert Index, droppt Duplikate
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV ist leer: {p}")

    time_col = None
    for cand in ["Date", "date", "time", "Time", "timestamp", "Timestamp"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)
    df.index.name = "time"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def coerce_series_numeric(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    return s2.dropna()


def infer_base_timestep_hours(idx: pd.DatetimeIndex) -> Optional[float]:
    """
    Versucht den typischen Zeitschritt in Stunden zu schätzen (median diff).
    """
    if len(idx) < 3:
        return None
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None
    median_dt = diffs.median()
    return float(median_dt / pd.Timedelta(hours=1))


def rolling_mean_centered(
    s: pd.Series,
    window: int,
    require_full_window: bool = True,
) -> pd.Series:
    """
    Zentrierter Rolling Mean ohne Rand-Artefakte.
    - window bezieht sich auf Samples der *aktuellen* Serie (raw oder aggregiert)
    - require_full_window=True => Ränder werden NaN (min_periods=window)
    """
    if window <= 1:
        return s.copy()

    minp = window if require_full_window else max(1, window // 2)
    return s.rolling(window=window, center=True, min_periods=minp).mean()


def expected_count_for_period(rule: str, base_hours: float, year: int) -> int:
    """
    Erwartete Mindestanzahl Samples pro Aggregationsperiode, um sie als "vollständig genug"
    zu akzeptieren.

    - Für '1D' und '1W' ist das eindeutig.
    - Für '1M' variieren Monatslängen; wir akzeptieren als Mindestschwelle 28 Tage.
    """
    samples_per_hour = 1.0 / base_hours
    if rule == "1D":
        return int(round(24 * samples_per_hour))
    if rule == "1W":
        return int(round(7 * 24 * samples_per_hour))
    if rule == "1M":
        return int(round(28 * 24 * samples_per_hour))  # robuste Untergrenze
    raise ValueError(f"Unsupported rule: {rule}")


def resample_mean_full_windows(
    s: pd.Series,
    rule: str,
    base_hours: float,
    year: int,
    min_fraction: float = 0.99,
) -> Tuple[pd.Series, int]:
    """
    Resample-Mittelwerte, aber setze Perioden mit zu wenigen Samples auf NaN.

    Returns:
      agg_mean, dropped_periods
    """
    mean = s.resample(rule).mean()
    cnt = s.resample(rule).count()

    expected_min = expected_count_for_period(rule, base_hours=base_hours, year=year)
    threshold = int(np.ceil(expected_min * float(min_fraction)))

    valid = cnt >= threshold
    dropped = int((~valid).sum())

    mean = mean.where(valid)
    return mean, dropped


def clip_to_full_year(s: pd.Series, year: int) -> pd.Series:
    """
    Schneidet die Serie strikt auf [year-01-01 00:00, year-12-31 23:59:59] zu.
    Bei stündlichen Daten ist das effektiv bis 23:00 am 31.12.
    """
    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    return s.loc[(s.index >= start) & (s.index <= end)]


def basic_stats(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame({"value": []})

    return pd.DataFrame(
        {
            "value": [
                float(s.min()),
                float(s.quantile(0.05)),
                float(s.mean()),
                float(s.median()),
                float(s.quantile(0.95)),
                float(s.max()),
                float(s.sum()),
                int(s.shape[0]),
            ]
        },
        index=["min", "p05", "mean", "median", "p95", "max", "sum", "n"],
    )


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="Demand Viewer", layout="wide")
st.title("Electricity Demand Viewer (non-hist) — robuste Aggregation ohne Rand-Drops")

with st.sidebar:
    st.header("Einstellungen")
    st.caption("Pfad-Konfiguration")
    base_dir = st.text_input("DEMAND_BASE_DIR", str(DEMAND_BASE_DIR))
    base_dir_p = Path(base_dir)

    years = find_available_years(base_dir_p)
    if not years:
        st.error(
            f"Keine Ordner im Format 'demand_YYYY' gefunden unter:\n{base_dir_p}\n\n"
            f"Erwartet: {base_dir_p}/demand_<year>/{DEMAND_FILENAME}"
        )
        st.stop()

    year = st.selectbox("Jahr", years, index=len(years) - 1)

    csv_path = demand_file_for_year(base_dir_p, int(year))
    st.write("Datei:", f"`{csv_path}`")

    if not csv_path.exists():
        st.error(f"Datei existiert nicht: {csv_path}")
        st.stop()

    with st.spinner("Lade Demand CSV..."):
        df = load_demand_csv(str(csv_path))

    cols = list(df.columns)
    if not cols:
        st.error("Keine Spalten (Länder) in der CSV gefunden.")
        st.stop()

    likely_countries = [c for c in cols if re.fullmatch(r"[A-Za-z]{2,3}", str(c))]
    default_col = likely_countries[0] if likely_countries else cols[0]
    country = st.selectbox("Land (Spalte)", cols, index=cols.index(default_col))

    st.divider()
    st.subheader("Darstellung")

    view_mode = st.radio(
        "Aggregation",
        ["Original (raw)", "Täglich (1D)", "Wöchentlich (1W)", "Monatlich (1M)"],
        index=0,
    )

    st.caption("Aggregation verwirft unvollständige Perioden durch Count-Check (setzt sie auf NaN).")
    min_fraction = st.slider(
        "Mindest-Abdeckung pro Periode",
        min_value=0.80,
        max_value=1.00,
        value=0.99,
        step=0.01,
        help="0.99 = mindestens 99% der erwarteten Samples; unvollständige Perioden werden NaN.",
    )

    enforce_full_year_clip = st.checkbox(
        "Serie vor Aggregation strikt auf das Kalenderjahr clippen",
        value=True,
        help="Empfohlen. Verhindert, dass Datenpunkte aus dem Folgejahr in den Dezember fallen oder umgekehrt.",
    )

    show_raw_points = st.checkbox(
        "Rohwerte als Punkte (nur Original, kann langsam sein)",
        value=False,
    )

    st.divider()
    st.subheader("Rolling Mean (optional)")
    add_roll = st.checkbox("Rolling Mean anzeigen", value=False)

    roll_window = st.selectbox(
        "Rolling Window (in Samples der aktuellen Aggregation)",
        [2, 5, 7, 14, 30],
        index=2,
    )

    require_full_window = st.checkbox(
        "Rolling Mean: nur volle Fenster (Ränder als NaN)",
        value=True,
    )

    st.divider()
    st.subheader("Export")
    include_processed = st.checkbox("Download: aktuelle Ansicht (nach Aggregation/Rolling)", value=True)


# =============================================================================
# Daten vorbereiten
# =============================================================================
raw_s = coerce_series_numeric(df[country]).rename(f"{country} demand")

if raw_s.empty:
    st.error(f"Spalte '{country}' enthält keine numerischen Werte nach Parsing.")
    st.stop()

base_hours = infer_base_timestep_hours(raw_s.index)
if base_hours is None:
    st.error("Konnte den Zeitschritt nicht bestimmen (zu wenige Datenpunkte).")
    st.stop()

if enforce_full_year_clip:
    raw_s = clip_to_full_year(raw_s, int(year))

if raw_s.empty:
    st.error("Nach dem Jahr-Clip sind keine Daten mehr vorhanden.")
    st.stop()

rule: Optional[str] = None
if view_mode == "Täglich (1D)":
    rule = "1D"
elif view_mode == "Wöchentlich (1W)":
    rule = "1W"
elif view_mode == "Monatlich (1M)":
    rule = "1M"

dropped_periods = 0
s = raw_s

if rule is not None:
    s_mean, dropped = resample_mean_full_windows(
        raw_s,
        rule=rule,
        base_hours=base_hours,
        year=int(year),
        min_fraction=float(min_fraction),
    )
    s = s_mean.rename(f"{country} demand ({rule} mean)")
    dropped_periods = dropped

roll_s = None
if add_roll:
    roll_s = rolling_mean_centered(
        s,
        window=int(roll_window),
        require_full_window=bool(require_full_window),
    ).rename(f"{country} demand (centered rolling {roll_window})")


# =============================================================================
# Layout
# =============================================================================
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader(f"Zeitreihe: {country} — {year}")

    if rule is not None and dropped_periods > 0:
        st.warning(
            f"Aggregation '{rule}': {dropped_periods} Periode(n) als unvollständig verworfen (NaN gesetzt). "
            f"(min_fraction={min_fraction:.2f})"
        )

    fig = plt.figure()
    ax = plt.gca()

    if show_raw_points and rule is None:
        ax.plot(s.index, s.values, marker=".", linestyle="None", markersize=1.5, label="raw")
    else:
        ax.plot(s.index, s.values, linewidth=1.2, label="demand")

    if roll_s is not None:
        ax.plot(roll_s.index, roll_s.values, linewidth=2.0, label="centered rolling mean")

    ax.set_xlabel("time")
    ax.set_ylabel("demand")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="best")

    st.pyplot(fig, clear_figure=True)

    st.caption("Zeitraumfilter (optional)")
    z1, z2 = st.columns(2)
    tmin = z1.date_input("Start", value=None)
    tmax = z2.date_input("Ende", value=None)

    if tmin is not None or tmax is not None:
        tmin_ts = pd.to_datetime(tmin) if tmin is not None else s.index.min()
        tmax_ts = (pd.to_datetime(tmax) + pd.Timedelta(days=1)) if tmax is not None else s.index.max()
        mask = (s.index >= tmin_ts) & (s.index <= tmax_ts)

        out = pd.DataFrame({"demand": s.loc[mask]})
        if roll_s is not None:
            out["rolling_mean"] = roll_s.loc[mask]

        st.write(f"Zeitraum: {out.index.min()} → {out.index.max()} (n={len(out)})")
        st.dataframe(out, use_container_width=True, height=260)

with c2:
    st.subheader("Checks / Kennzahlen")

    st.markdown("**Rohdaten (nach Clip):**")
    st.write(
        f"Start: `{raw_s.index.min()}`  \n"
        f"Ende: `{raw_s.index.max()}`  \n"
        f"Median Δt: `{pd.to_timedelta(base_hours, unit='h')}`  \n"
        f"n: `{len(raw_s)}`"
    )

    st.divider()

    st.markdown("**Stats (roh):**")
    st.dataframe(basic_stats(raw_s), use_container_width=True)

    st.markdown("**Stats (aktuelle Ansicht):**")
    st.dataframe(basic_stats(s), use_container_width=True)

    st.divider()
    st.subheader("Daten-Preview")
    st.dataframe(df.head(20), use_container_width=True, height=220)


# =============================================================================
# Download
# =============================================================================
if include_processed:
    out_df = pd.DataFrame(index=s.index)
    out_df.index.name = "time"
    out_df["demand"] = s
    if roll_s is not None:
        out_df["rolling_mean"] = roll_s

    csv_bytes = out_df.to_csv().encode("utf-8")
    st.download_button(
        label=f"Download CSV: {country} {year} ({view_mode})",
        data=csv_bytes,
        file_name=f"demand_{year}_{country}_{view_mode.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        mime="text/csv",
    )
else:
    csv_bytes = raw_s.to_frame(name="demand").to_csv().encode("utf-8")
    st.download_button(
        label=f"Download CSV: {country} {year} (raw)",
        data=csv_bytes,
        file_name=f"demand_{year}_{country}_raw.csv",
        mime="text/csv",
    )
