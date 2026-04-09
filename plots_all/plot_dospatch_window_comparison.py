#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vergleicht Dispatch-Zeitreihen mehrerer Netzwerke/Dispatches
- einmal ueber den kompletten Zeitraum
- einmal ueber dispatch-spezifische Dunkelflautenfenster aus master_config

Funktioniert fuer normale und ARO-Runs.
Bei ARO-Runs koennen explizit Worst-Case-Dispatches bzw. alle Szenarien geladen werden.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from master_config import MasterConfig
from run_loader import RunLoader, RunSpec

COMPARE_SPECS = [
    # {"key": "run-a", "label": "Basis"},
    # {"key": "aro-run", "label": "ARO", "aro_include_scenarios": True},
]
COMPARE_COUNTRIES = ["ALL"]
OUT_DIR = None

GEN_CARRIERS = ["onwind", "offwind-ac", "offwind-dc", "offwind-float", "solar", "solar rooftop", "solar-hsat", "ror", "hydro", "nuclear", "biomass"]
LINK_DISCHARGE = ["battery discharger", "H2 Fuel Cell"]
STORE_CARRIERS = ["PHS", "battery", "H2 storage"]


def _slug(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(s))


def _select_spec_tag(spec: RunSpec, prefer_worst_case: bool = True):
    tags = list(spec.tags)
    if not tags:
        return None
    if prefer_worst_case:
        for t in tags:
            ts = str(t)
            if 'worst' in ts.lower():
                return t
    for t in tags:
        if str(t) != 'robust':
            return t
    return tags[0]


def _filter_cols(cols, country: str):
    if country == 'ALL':
        return list(cols)
    out = []
    for c in cols:
        s = str(c)
        if s.startswith(country) or f' {country}' in s or s[:2] == country:
            out.append(c)
    return out


def _series_or_zero(df, cols, index):
    cols2 = [c for c in cols if c in df.columns]
    if not cols2:
        return pd.Series(0.0, index=index)
    return df[cols2].sum(axis=1)


def extract_balance(n: pypsa.Network, country: str) -> pd.DataFrame:
    idx = n.snapshots
    gen_mask = n.generators.carrier.astype(str).isin(GEN_CARRIERS)
    gen_cols = _filter_cols(n.generators.index[gen_mask], country)
    generation = _series_or_zero(n.generators_t.p, gen_cols, idx)

    link_mask = n.links.carrier.astype(str).isin(LINK_DISCHARGE)
    link_cols = _filter_cols(n.links.index[link_mask], country)
    storage_out = _series_or_zero(n.links_t.p1.abs() if hasattr(n.links_t, 'p1') else pd.DataFrame(index=idx), link_cols, idx)

    load_cols = _filter_cols(n.loads.index, country)
    demand = _series_or_zero(n.loads_t.p_set if hasattr(n.loads_t, 'p_set') else pd.DataFrame(index=idx), load_cols, idx)

    return pd.DataFrame({
        'generation_gw': generation / 1e3,
        'storage_out_gw': storage_out / 1e3,
        'demand_gw': demand / 1e3,
    }, index=idx)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    return {
        'generation_twh': float(df['generation_gw'].sum()),
        'storage_out_twh': float(df['storage_out_gw'].sum()),
        'demand_twh': float(df['demand_gw'].sum()),
        'generation_mean_gw': float(df['generation_gw'].mean()),
        'demand_mean_gw': float(df['demand_gw'].mean()),
        'hours': float(len(df)),
    }


def plot_lines(series_map: Dict[str, pd.DataFrame], title: str, outpath: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    keys = list(series_map.keys())
    for label in keys:
        df = series_map[label]
        axes[0].plot(df.index, df['generation_gw'], label=label)
        axes[1].plot(df.index, df['storage_out_gw'], label=label)
        axes[2].plot(df.index, df['demand_gw'], label=label)
    axes[0].set_ylabel('Generation [GW]')
    axes[1].set_ylabel('Storage out [GW]')
    axes[2].set_ylabel('Demand [GW]')
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def plot_bars(full_summary, window_summary, title: str, outpath: Path):
    labels = list(full_summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    metrics = [('generation_twh', 'Generation [TWh]'), ('demand_twh', 'Demand [TWh]')]
    for ax, (metric, ttl) in zip(axes, metrics):
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], [full_summary[k][metric] for k in labels], width=0.4, label='full')
        ax.bar([i + 0.2 for i in x], [window_summary[k][metric] for k in labels], width=0.4, label='Dunkelflaute')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_title(ttl)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def build_specs(master: MasterConfig, cli_runs: Optional[List[str]]) -> List[Tuple[str, RunSpec, str]]:
    raw_specs = []
    if cli_runs:
        for item in cli_runs:
            if ':' in item:
                key, label = item.split(':', 1)
            else:
                key, label = item, item
            raw_specs.append({'key': key, 'label': label, 'aro_include_scenarios': True})
    else:
        raw_specs = COMPARE_SPECS

    loader = RunLoader(master)
    out = []
    for d in raw_specs:
        spec = loader.load(
            d['key'],
            label=d.get('label'),
            aro_include_scenarios=d.get('aro_include_scenarios', False),
        )
        tag = _select_spec_tag(spec, prefer_worst_case=True)
        if tag is None:
            continue
        out.append((d.get('label', d['key']), spec, str(tag)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--countries', nargs='*', default=None)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    master = MasterConfig()
    specs = build_specs(master, args.runs)
    countries = args.countries or COMPARE_COUNTRIES
    outdir = Path(args.outdir or OUT_DIR or (Path(master.plots_base) / 'dispatch_window_comparison'))
    outdir.mkdir(parents=True, exist_ok=True)

    for country in countries:
        all_full = {}
        all_win = {}
        full_summary = {}
        window_summary = {}

        for label, spec, tag in specs:
            n = spec.get_network(tag)
            if n is None:
                continue
            df = extract_balance(n, country)
            all_full[label] = df
            full_summary[label] = summarize(df)

            window = master.get_dispatch_window(tag, spec.run_key)
            start = pd.Timestamp(window['start'])
            end = pd.Timestamp(window['end'])
            dfw = df.loc[(df.index >= start) & (df.index <= end)].copy()
            all_win[label] = dfw
            window_summary[label] = summarize(dfw if not dfw.empty else df.iloc[0:0].assign(generation_gw=[], storage_out_gw=[], demand_gw=[]))

        if not all_full:
            continue

        plot_lines(all_full, f'{country}: Dispatch comparison (full period)', outdir / f'{_slug(country)}_dispatch_full.png')
        plot_lines(all_win, f'{country}: Dispatch comparison (dispatch-specific Dunkelflaute)', outdir / f'{_slug(country)}_dispatch_dunkelflaute.png')
        plot_bars(full_summary, window_summary, f'{country}: Full vs Dunkelflaute summary', outdir / f'{_slug(country)}_dispatch_summary.png')


if __name__ == '__main__':
    main()