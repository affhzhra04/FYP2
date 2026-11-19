"""Fuel-theft detector validation by injection.

The telemetry has no labelled theft, so we create ground truth: take real moments where the vehicle is parked
with the ignition off and the ping arrived within 15 minutes of the previous one (in practice the first report after
the engine is switched off), lower the fuel level by a known volume from that ping to the end of the session, then check
that the detector finds each injected drop.

Scope of the result: the detector can only see drops it can time-stamp. Parked vehicles otherwise report once an hour and,
over such gaps, the fuel sensor moves up and down by several litres on its own, so drops there cannot be told from noise.

Run from the project root:   python scripts/validate_fuel_injection.py [--n 60] [--litres 8] [--seed 7]
Writes docs/results/fuel_injection_seed<N>.csv and prints recall plus any extra fuel events caused by the injection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from database.db_manager import DEFAULT_DB_PATH
from src import dash_core as C
from src import risk_engine as R

FUEL_KINDS = ["fuel_theft", "fuel_off_route"]


def dense_parked_runs(prep: pd.DataFrame, min_len: int = 1):
    """(session_id, row index) of every parked, ignition-off ping that arrived at most 15 minutes after the one
    before it and is followed by another ping in the same session (needed for the persistence check)."""
    dense = (prep["speed"] < 1) & (prep["ignition"] == 0) & (prep["dt"] <= 900)
    run = (dense != dense.shift()).cumsum()
    out = []
    for _, g in prep[dense].groupby([prep["session_id"][dense], run[dense]]):
        i = g.index[0]
        if len(g) >= min_len and i + 1 < len(prep) and prep.loc[i + 1, "session_id"] == g["session_id"].iloc[0]:
            out.append((g["session_id"].iloc[0], i))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--litres", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    a = ap.parse_args()

    pings = C.read_pings(a.db).sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    pings["timestamp"] = pd.to_datetime(pings["timestamp"])
    base_events = R.detect_all(pings)[1]
    base_fuel = base_events[base_events["kind"].isin(FUEL_KINDS)]

    rng = np.random.default_rng(a.seed)
    prep = R.prepare(pings)
    assert len(prep) == len(pings) and (prep["timestamp"].to_numpy() == pings["timestamp"].to_numpy()).all(), \
        "row order changed during preparation"
    cands = dense_parked_runs(prep)
    print(f"{len(cands)} candidate parked runs with dense pings found.")
    seen, picks = set(), []
    for i in rng.permutation(len(cands)):
        sid, idx = cands[i]
        if sid not in seen:                      # one injection per session keeps matching unambiguous
            seen.add(sid)
            picks.append((sid, idx))
        if len(picks) == a.n:
            break

    mod = pings.copy()
    truth = []
    for sid, idx in picks:
        end = mod.index[mod["vehicle_id"] == sid].max()
        mod.loc[idx:end, "fuel_level"] -= a.litres
        truth.append(dict(session_id=sid, t=mod.loc[idx, "timestamp"]))

    ev = R.detect_all(mod)[1]
    fuel = ev[ev["kind"].isin(FUEL_KINDS)]
    rows, hit = [], 0
    for tr in truth:
        g = fuel[(fuel["session_id"] == tr["session_id"]) &
                 (fuel["start"] >= tr["t"] - pd.Timedelta(seconds=60)) &
                 (fuel["start"] <= tr["t"] + pd.Timedelta(minutes=15))]
        ok = len(g) > 0
        hit += int(ok)
        rows.append(dict(session_id=tr["session_id"], injected_at=tr["t"], litres=a.litres, detected=ok,
                         detected_litres=float(g["magnitude"].max()) if ok else np.nan))
    res = pd.DataFrame(rows)
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"fuel_injection_seed{a.seed}.csv", index=False)
    extra = len(fuel) - len(base_fuel) - hit
    print(f"Injected {len(truth)} drops of {a.litres:g} L; detected {hit} (recall {hit / max(1, len(truth)):.0%}).")
    print(f"Fuel events before injection: {len(base_fuel)}; after: {len(fuel)}; unmatched extra events: {extra}.")
    print(f"Wrote {out}")
    print("Scope: drops at engine-off moments with dense pings; hourly-heartbeat gaps are noise-dominated.")


if __name__ == "__main__":
    main()
