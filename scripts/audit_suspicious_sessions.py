"""Find sessions whose score may be driven by parked time rather than driving (like session 1_29_9).

Session 1_29_9 covers 53 km in 24 h and scores 0.0 because five long "unauthorized stops" (66 to 674 min) dominate
a short distance. A long stop at an unlisted place can be a real problem (unplanned stop) or an ordinary one (overnight
parking, loading, a customer site the fleet has not visited often). This script lists every session where stops dominate
the score so a person can check them, and says why each was listed.

Run from the project root:   python scripts/audit_suspicious_sessions.py
Writes docs/results/suspicious_sessions.csv and prints the top rows.
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

LONG_STOP_MIN = 240.0     # a stop this long is more likely parking than a delivery stop
SHARE_DOMINANT = 0.6      # stops account for at least this share of the session's risk points
LOW_KM = 100.0            # short distance makes a few stops dominate the per-100-km score
LOW_SCORE = 25.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "suspicious_sessions.csv"))
    a = ap.parse_args()

    pings = C.read_pings(a.db)
    pings["timestamp"] = pd.to_datetime(pings["timestamp"])
    bundle = C.build_bundle(pings)
    scores = C.session_scores(bundle, R.DEFAULT_WEIGHTS)
    ev = bundle.events

    rows = []
    for sid, g in ev.groupby("session_id"):
        stops = g[g["kind"] == "unauthorized_stop"]
        if stops.empty:
            continue
        audit = C.session_audit(bundle, sid, R.DEFAULT_WEIGHTS)
        pts = audit.items.groupby("kind")["Points"].sum()
        total = float(pts.sum())
        share = float(pts.get("unauthorized_stop", 0.0) / total) if total > 0 else 0.0
        rows.append(dict(session_id=sid, vehicle_id=int(g["vehicle_id"].iloc[0]),
                         km=round(audit.meta["km"], 1), score=round(audit.score, 1), tier=audit.tier,
                         stops=len(stops), stop_minutes=round(float(stops["magnitude"].sum()), 0),
                         longest_stop_min=round(float(stops["magnitude"].max()), 0),
                         share_of_risk_from_stops=round(share, 2)))
    out = pd.DataFrame(rows)
    if out.empty:
        print("No unauthorized stops found.")
        return

    why = []
    for _, r in out.iterrows():
        w = []
        if r["share_of_risk_from_stops"] >= SHARE_DOMINANT:
            w.append(f"stops are {r['share_of_risk_from_stops']:.0%} of the risk")
        if r["longest_stop_min"] >= LONG_STOP_MIN:
            w.append(f"a stop of {r['longest_stop_min']:.0f} min (likely overnight parking)")
        if r["km"] < LOW_KM and r["score"] < LOW_SCORE:
            w.append(f"only {r['km']:.0f} km, so a few stops dominate the per-100-km score")
        why.append("; ".join(w))
    out["why_check"] = why
    out = out[out["why_check"] != ""].sort_values(["share_of_risk_from_stops", "stop_minutes"], ascending=False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} of {scores['session_id'].nunique()} sessions need a look. Saved to {a.out}")
    print(out.head(15).to_string(index=False))
    both = out[(out["km"] < LOW_KM) & (out["score"] < LOW_SCORE)]
    print(f"\n{len(both)} are short, low-score sessions like 1_29_9.")


if __name__ == "__main__":
    main()
