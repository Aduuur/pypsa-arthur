# compare_utils.py
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# Generic helpers
# =============================================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, which="major", alpha=0.25)


def warn(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


# =============================================================================
# CSV helpers
# =============================================================================
def robust_read_csv_timeindexed(fp: Path, year: int) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp)
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in {"snapshot", "snapshots", "time", "datetime", "date", "timestamp"}:
            time_col = c
            break

    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce")
        num = df.drop(columns=[time_col]).select_dtypes(include=[np.number])
        num.index = pd.DatetimeIndex(t)
    else:
        df2 = pd.read_csv(fp, index_col=0)
        df2 = df2.drop(columns=[c for c in df2.columns if c.lower().startswith("unnamed")], errors="ignore")
        num = df2.select_dtypes(include=[np.number])
        num.index = pd.to_datetime(df2.index, errors="coerce")

    num = num[~num.index.isna()].sort_index()
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns in {fp}")
    return num


_COUNTRY_RE = re.compile(r"^\s*([A-Z]{2})\b")


def country_from_col(col: str) -> Optional[str]:
    m = _COUNTRY_RE.match(col)
    return m.group(1) if m else None


def drop_countries(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    ex = set(exclude)
    keep = []
    for c in df.columns:
        cc = country_from_col(c)
        if cc is None or cc not in ex:
            keep.append(c)
    return df[keep]


def group_cols_by_country(cols: List[str], exclude: Iterable[str]) -> Dict[str, List[str]]:
    ex = set(exclude)
    out: Dict[str, List[str]] = {}
    for c in cols:
        cc = country_from_col(c)
        if cc and cc not in ex:
            out.setdefault(cc, []).append(c)
    return out


def pick_units_from_cols(cols: List[str]) -> Optional[str]:
    m = re.search(r"\[(.+?)\]|\((.+?)\)", " ".join(cols))
    if not m:
        return None
    return (m.group(1) or m.group(2) or "").strip() or None


def ylabel(base: str, units: Optional[str]) -> str:
    return f"{base} [{units}]" if units else base


# =============================================================================
# PyPSA helpers
# =============================================================================
def try_import_pypsa():
    try:
        import pypsa  # type: ignore
        return pypsa
    except Exception as e:
        warn(f"pypsa not available ({e}). Network-based analyses will be skipped.")
        return None


def _is_resources_path(p: Path) -> bool:
    s = str(p).lower()
    return "/resources/" in s or "\\resources\\" in s


def _is_results_path(p: Path) -> bool:
    s = str(p).lower()
    return "/results/" in s or "\\results\\" in s


def _extract_run_name(p: Path) -> Optional[str]:
    """
    Try to find the run name after 'resources' or 'results', e.g.
    .../resources/compare-era5/networks/... -> compare-era5
    """
    parts = [x for x in p.parts]
    low = [x.lower() for x in parts]
    for key in ("resources", "results"):
        if key in low:
            i = low.index(key)
            if i + 1 < len(parts):
                return parts[i + 1]
    return None


def _map_to_results_root(run_dir: Path) -> Optional[Path]:
    """
    If run_dir points into resources/.../compare-XYZ/...,
    map to .../results/compare-XYZ.

    If run_dir already in results, return .../results/compare-XYZ (normalized).
    """
    run_dir = run_dir.resolve()
    run_name = _extract_run_name(run_dir)
    if run_name is None:
        return None

    # Find project root by walking up until we hit 'resources'/'results'
    parts = list(run_dir.parts)
    low = [x.lower() for x in parts]
    if "resources" in low:
        i = low.index("resources")
        root = Path(*parts[:i])  # everything before resources
        return (root / "results" / run_name).resolve()
    if "results" in low:
        i = low.index("results")
        root = Path(*parts[:i])
        return (root / "results" / run_name).resolve()

    return None


def _has_results_tables(n) -> bool:
    """
    Heuristic: network contains evidence of being solved / having results.
    """
    try:
        if hasattr(n, "buses_t") and hasattr(n.buses_t, "marginal_price"):
            mp = n.buses_t.marginal_price
            if isinstance(mp, pd.DataFrame) and mp.shape[1] > 0:
                return True
        if hasattr(n, "generators_t") and hasattr(n.generators_t, "p"):
            p = n.generators_t.p
            if isinstance(p, pd.DataFrame) and p.shape[1] > 0:
                return True
        if hasattr(n, "links_t") and hasattr(n.links_t, "p0"):
            p0 = n.links_t.p0
            if isinstance(p0, pd.DataFrame) and p0.shape[1] > 0:
                return True
        if hasattr(n, "lines_t") and hasattr(n.lines_t, "p0"):
            p0 = n.lines_t.p0
            if isinstance(p0, pd.DataFrame) and p0.shape[1] > 0:
                return True
    except Exception:
        return False
    return False


def _candidate_network_files(results_root: Path) -> List[Path]:
    """
    Only look where solved networks should be: results_root/**/networks/*.nc
    """
    cands = []
    cands += list(results_root.glob("networks/*.nc"))
    cands += list(results_root.glob("**/networks/*.nc"))
    # dedup
    seen: set[Path] = set()
    uniq: List[Path] = []
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)

    # Prefer newest, and names that look like final solved outputs
    def score(p: Path) -> Tuple[int, float, int]:
        name = p.name.lower()
        s = 0
        if "solved" in name or "post" in name or "lopf" in name or "opf" in name:
            s -= 20
        if "base" in name:
            s -= 5
        if "elec" in name:
            s -= 2
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        plen = len(str(p))
        return (s, -mtime, plen)

    uniq.sort(key=score)
    return uniq


def load_first_network(run_dir: Path):
    """
    Loads the *correct* solved network for a run.
    - If run_dir is in resources/, we map to results/<run>/...
    - We search only in results/<run>/**/networks/*.nc
    - If nothing found, we DO NOT silently fall back to resources (unless enabled below).
    """
    pypsa = try_import_pypsa()
    if pypsa is None:
        return None, None

    run_dir = Path(run_dir)

    # Toggle: allow fallback to resources (not recommended; will lead to empty plots)
    ALLOW_RESOURCES_FALLBACK = False

    results_root = _map_to_results_root(run_dir)
    if results_root is None:
        warn(
            f"Could not map run_dir to a results/<run> root (run_dir={run_dir}). "
            "Please point CompareConfig.run_dirs to /.../results/<run>."
        )
        return None, None

    if not results_root.exists():
        warn(
            f"Known run name but results directory does not exist: {results_root}. "
            "Did you run the snakemake/solve for this run?"
        )
        return None, None

    cands = _candidate_network_files(results_root)
    if not cands:
        warn(f"No network .nc found under {results_root}/**/networks/.")
        return None, None

    # Prefer a network that has results tables
    first_plausible: Tuple[Optional[object], Optional[Path]] = (None, None)
    for fp in cands[:80]:
        try:
            n = pypsa.Network(fp)
        except Exception:
            continue

        if getattr(n, "buses", None) is None or len(n.buses) == 0:
            continue

        if first_plausible[0] is None:
            first_plausible = (n, fp)

        if _has_results_tables(n):
            return n, fp

    # No results tables found: still return plausible candidate, but warn loudly.
    if first_plausible[0] is not None:
        n, fp = first_plausible
        warn(
            f"Loaded a network from results but did not detect result tables "
            f"(no buses_t.marginal_price / generators_t.p / flows). File: {fp}. "
            "This typically means it is a base network (not solved) or results were not stored."
        )
        return n, fp

    # Optional resources fallback (not recommended)
    if ALLOW_RESOURCES_FALLBACK:
        warn("Falling back to resources/ search (will likely produce empty plots).")
        # very last resort: old behavior
        cands2 = list(run_dir.glob("**/networks/*.nc"))
        for fp in cands2[:50]:
            try:
                n = pypsa.Network(fp)
                if getattr(n, "buses", None) is not None and len(n.buses) > 0:
                    return n, fp
            except Exception:
                continue

    warn(f"Found candidates but none could be loaded as PyPSA network under {results_root}.")
    return None, None


def snapshot_weight(n) -> pd.Series:
    """
    Robust: returns snapshot weights w_t (typically hours) aligned to n.snapshots.
    """
    snaps = getattr(n, "snapshots", None)
    if snaps is None or len(snaps) == 0:
        return pd.Series(dtype=float)

    sw = getattr(n, "snapshot_weightings", None)
    if sw is None or len(sw) == 0:
        return pd.Series(1.0, index=snaps)

    if isinstance(sw, pd.DataFrame):
        for col in ["generators", "objective", "stores"]:
            if col in sw.columns:
                return pd.to_numeric(sw[col], errors="coerce").reindex(snaps).fillna(1.0)
        return pd.to_numeric(sw.iloc[:, 0], errors="coerce").reindex(snaps).fillna(1.0)

    try:
        return pd.Series(pd.to_numeric(sw, errors="coerce"), index=snaps).fillna(1.0)
    except Exception:
        return pd.Series(1.0, index=snaps)


def month_mask(index: pd.DatetimeIndex, year: int, month: int) -> np.ndarray:
    return (index.year == year) & (index.month == month)


def safe_group_sum(values: pd.Series, by: pd.Series) -> pd.Series:
    if values is None or len(values) == 0:
        return pd.Series(dtype=float)
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    k = by.reindex(v.index) if by is not None else pd.Series("unknown", index=v.index)
    k = k.astype(str).fillna("unknown")
    df = pd.DataFrame({"v": v, "k": k})
    s = df.groupby("k")["v"].sum().sort_values(ascending=False)
    s = s[s != 0.0]
    return s
