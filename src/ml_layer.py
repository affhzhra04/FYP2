"""Machine-learning layer on top of the rule-based risk engine.

Two models, both trained on per-session *kinematic* features (speed profile, driving
roughness, idle share, night share, turning, stationary fuel loss, spatial extent):

1. Isolation Forest (unsupervised): anomaly score per session, no labels needed.
   Used to cross-check the rules and to surface sessions the rules rated as safe.
2. Random Forest regressor (supervised): predicts the rule-based risk rate per 100 km
   from the kinematics alone and is evaluated with leave-one-vehicle-out cross
   validation, i.e. on a vehicle the model has never seen. A gradient boosting model and
   a median baseline are reported next to it.

Honest limits (state them in the report): the supervised target is produced by the
rules, so this measures how well behaviour patterns reproduce and generalise the rule
score; it is not accuracy against ground-truth crashes. No external labels exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import risk_engine as R

FEATURES: List[str] = [
    "km", "moving_h", "speed_mean", "speed_p90", "speed_p99", "speed_max", "speed_std",
    "fast_share", "accel_mean", "accel_p99", "idle_share", "night_share", "turn_rate",
    "fuel_drop_stationary", "extent_km",
]
FEATURE_LABEL: Dict[str, str] = {
    "km": "Distance driven", "moving_h": "Hours moving", "speed_mean": "Mean speed (moving)",
    "speed_p90": "Speed 90th percentile", "speed_p99": "Speed 99th percentile", "speed_max": "Top speed",
    "speed_std": "Speed variability", "fast_share": "Share of moving time above 80 km/h",
    "accel_mean": "Mean acceleration magnitude", "accel_p99": "Hard acceleration (99th pct)",
    "idle_share": "Idle share of session", "night_share": "Moving at night (22:00-05:00)",
    "turn_rate": "Turning rate", "fuel_drop_stationary": "Fuel dropped while stationary (L)",
    "extent_km": "Spatial extent of trip",
}
IF_FEATURES: List[str] = ["speed_mean", "speed_p90", "speed_p99", "speed_std", "fast_share", "accel_mean",
                          "accel_p99", "idle_share", "night_share", "turn_rate", "fuel_drop_stationary", "km",
                          "extent_km"]
MIN_ACTIVE_KM = 50.0
MIN_PINGS = 20
TIERS = ["Critical", "High Risk", "Moderate Risk", "Low Risk", "Excellent"]
REASON_PHRASE: Dict[str, str] = {
    "speed_mean": "high average speed", "speed_p90": "high speeds", "speed_p99": "very high top speeds",
    "speed_std": "erratic speed", "fast_share": "long time at high speed", "accel_mean": "rough acceleration",
    "accel_p99": "hard acceleration", "idle_share": "long idle time", "night_share": "night-time driving",
    "turn_rate": "frequent sharp turns", "fuel_drop_stationary": "fuel dropped while parked",
    "km": "unusually long distance", "extent_km": "unusually wide area covered",
}


# --------------------------------------------------------------------- features
def session_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of kinematic features per session from a prepared ping frame."""
    d = df.copy()
    dt = d["dt"].where((d["dt"] > 0) & (d["dt"] <= 300), 0.0).fillna(0.0)   # cap gaps at 5 min
    d["_dt"] = dt
    moving = d["speed"] > 5
    d["_mov_dt"] = dt.where(moving, 0.0)
    d["_fast_dt"] = dt.where(d["speed"] > 80, 0.0)
    d["_idle_dt"] = dt.where((d["speed"] < 1) & (d["ignition"] == 1), 0.0)
    hour = d["timestamp"].dt.hour
    d["_night_dt"] = d["_mov_dt"].where((hour >= 22) | (hour < 5), 0.0)
    ok = (d["dt"] > 0) & (d["dt"] <= 15) & moving
    d["_acc"] = (d["dv"].abs() / 3.6 / d["dt"]).where(ok)                     # m/s^2
    hd = d["heading"].astype(float)
    dh = (hd.groupby(d["session_id"]).diff().abs() % 360)
    dh = np.minimum(dh, 360 - dh)
    d["_turn"] = (dh / d["dt"]).where((d["dt"] > 0) & (d["dt"] <= 60) & moving)
    d["_spd_mov"] = d["speed"].where(moving)
    stat_drop = (d["dl"] < 0) & (d["speed"] < 1) & (d["ignition"] == 0) & (d["dt"] <= 900)
    d["_fdrop"] = (-d["dl"]).where(stat_drop, 0.0).fillna(0.0)

    g = d.groupby("session_id", sort=False)
    out = pd.DataFrame({
        "vehicle_id": g["vehicle_id"].first(),
        "pings": g.size(),
        "km": g["step_km"].sum(),
        "hours": g["_dt"].sum() / 3600.0,
        "moving_h": g["_mov_dt"].sum() / 3600.0,
        "speed_mean": g["_spd_mov"].mean(),
        "speed_p90": g["_spd_mov"].quantile(0.90),
        "speed_p99": g["_spd_mov"].quantile(0.99),
        "speed_max": g["speed"].max(),
        "speed_std": g["_spd_mov"].std(),
        "accel_mean": g["_acc"].mean(),
        "accel_p99": g["_acc"].quantile(0.99),
        "turn_rate": g["_turn"].mean(),
        "fuel_drop_stationary": g["_fdrop"].sum(),
        "lat_span": g["latitude"].agg(lambda s: s.max() - s.min()),
        "lon_span": g["longitude"].agg(lambda s: s.max() - s.min()),
    })
    tot = out["hours"].replace(0, np.nan)
    mov = out["moving_h"].replace(0, np.nan)
    out["fast_share"] = (g["_fast_dt"].sum() / 3600.0) / mov
    out["idle_share"] = (g["_idle_dt"].sum() / 3600.0) / tot
    out["night_share"] = (g["_night_dt"].sum() / 3600.0) / mov
    out["extent_km"] = np.hypot(out["lat_span"] * 111.0, out["lon_span"] * 111.0 * 0.995)
    out = out.drop(columns=["lat_span", "lon_span", "hours"])
    out[FEATURES] = out[FEATURES].fillna(0.0)
    return out.reset_index()


def _reasons(Z: np.ndarray) -> List[str]:
    """Plain-language reason per row: the (up to) two features that stand out most from other trips."""
    zdf = pd.DataFrame(Z, columns=IF_FEATURES)
    out = []
    for _, row in zdf.iterrows():
        top = row[row > 1.5].sort_values(ascending=False).head(2).index
        out.append(", ".join(REASON_PHRASE[c] for c in top) or "overall pattern differs from other trips")
    return out


# ---------------------------------------------------------------------- results
@dataclass
class MLResults:
    table: pd.DataFrame               # per session: features, rule score/tier, ML prediction, anomaly
    metrics: pd.DataFrame             # model comparison
    importance: pd.DataFrame          # feature importance (Random Forest)
    per_vehicle: pd.DataFrame         # leave-one-vehicle-out result for each held-out vehicle
    agreement: Dict[str, float]       # isolation forest vs rules
    confusion: pd.DataFrame           # rule tier (rows) vs ML-predicted tier (columns), active sessions
    n_active: int
    n_total: int


def _tier_index(score: pd.Series) -> pd.Series:
    return score.map(R.tier).map({t: i for i, t in enumerate(TIERS)})


def _score_from_log_rate(pred_log: np.ndarray, r0: float) -> np.ndarray:
    return 100.0 * np.exp(-np.expm1(np.clip(pred_log, 0, None)) / r0)


def run_ml(sessions_scored: pd.DataFrame, feats: pd.DataFrame, r0: float = 10.0,
           seed: int = 0, n_trees: int = 300) -> MLResults:
    """``sessions_scored``: output of ``dash_core.session_scores`` (default weights).

    Imports scikit-learn lazily so the rest of the app works without it.
    """
    from scipy.stats import spearmanr
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import RobustScaler

    t = feats.merge(sessions_scored[["session_id", "score", "tier", "risk", "events"]], on="session_id", how="inner")
    t = t[t["pings"] >= MIN_PINGS].reset_index(drop=True)
    t["target"] = np.log1p(t["risk"].astype(float))
    active = (t["km"] >= MIN_ACTIVE_KM).to_numpy()
    X = t[FEATURES].to_numpy(float)
    y = t["target"].to_numpy(float)
    groups = t["vehicle_id"].to_numpy()

    def make(name):
        if name == "Random Forest":
            return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        if name == "Gradient Boosting":
            return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=200, random_state=seed)
        return DummyRegressor(strategy="median")

    names = ["Median baseline", "Gradient Boosting", "Random Forest"]
    preds = {n: np.zeros(len(t)) for n in names}
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        for n in names:
            m = make(n).fit(X[tr], y[tr])
            preds[n][te] = m.predict(X[te])

    true_score = t["score"].to_numpy(float)
    true_tier = _tier_index(t["score"]).to_numpy()
    rows = []
    for n in names:
        sc = _score_from_log_rate(preds[n], r0)
        tr_ = _tier_index(pd.Series(sc)).to_numpy()
        for label, mask in (("All sessions", np.ones(len(t), bool)), (f"Active sessions (>= {MIN_ACTIVE_KM:.0f} km)", active)):
            if mask.sum() < 3:
                continue
            rho = spearmanr(preds[n][mask], y[mask]).correlation if np.std(preds[n][mask]) > 0 else 0.0
            rows.append(dict(Model=n, Subset=label, Sessions=int(mask.sum()),
                             MAE_DSS=float(np.mean(np.abs(sc[mask] - true_score[mask]))),
                             Spearman=float(0.0 if np.isnan(rho) else rho),
                             Tier_exact=float(np.mean(tr_[mask] == true_tier[mask])),
                             Tier_within1=float(np.mean(np.abs(tr_[mask] - true_tier[mask]) <= 1))))
    metrics = pd.DataFrame(rows)

    rf_pred = preds["Random Forest"]
    t["ml_score"] = _score_from_log_rate(rf_pred, r0)
    t["ml_tier"] = t["ml_score"].map(R.tier)

    full_rf = make("Random Forest").fit(X, y)
    imp = pd.DataFrame({"feature": FEATURES, "importance": full_rf.feature_importances_})
    imp["label"] = imp["feature"].map(FEATURE_LABEL)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    pv = []
    for v in sorted(np.unique(groups), key=lambda x: (len(x), x)):
        mk = (groups == v) & active
        if mk.sum() >= 3:
            rho = spearmanr(rf_pred[mk], y[mk]).correlation
            pv.append(dict(Vehicle=v, Active_sessions=int(mk.sum()),
                           MAE_DSS=float(np.mean(np.abs(t["ml_score"][mk] - true_score[mk]))),
                           Spearman=float(0.0 if np.isnan(rho) else rho)))
    per_vehicle = pd.DataFrame(pv)

    # Isolation Forest on active sessions (kinematics only, no rule output as input)
    feat_for_if = IF_FEATURES
    t["anomaly"] = np.nan
    t["reason"] = ""
    agree = dict(spearman=np.nan, top_overlap=np.nan, n_ml_only=0)
    if active.sum() >= 20:
        Z = RobustScaler().fit_transform(np.log1p(np.abs(t.loc[active, feat_for_if].to_numpy(float))))
        iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(Z)
        raw = -iso.score_samples(Z)
        pct = pd.Series(raw).rank(pct=True).to_numpy() * 100.0
        t.loc[active, "anomaly"] = pct
        reasons = _reasons(Z)
        t.loc[active, "reason"] = reasons
        rho = spearmanr(pct, t.loc[active, "risk"]).correlation
        k = max(3, int(0.1 * active.sum()))
        top_ml = set(t.loc[active].iloc[np.argsort(-pct)[:k]]["session_id"])
        top_rule = set(t.loc[active].sort_values("risk", ascending=False).head(k)["session_id"])
        ml_only = t[active & (t["anomaly"] >= 90) & t["tier"].isin(["Excellent", "Low Risk"])]
        agree = dict(spearman=float(rho), top_overlap=len(top_ml & top_rule) / k, n_ml_only=int(len(ml_only)))
    t["ml_only_flag"] = (t["anomaly"] >= 90) & t["tier"].isin(["Excellent", "Low Risk"])

    conf = pd.crosstab(pd.Categorical(t.loc[active, "tier"], TIERS, ordered=True),
                       pd.Categorical(t.loc[active, "ml_tier"], TIERS, ordered=True), dropna=False)
    conf.index.name, conf.columns.name = "Rule-based tier", "ML-predicted tier"
    return MLResults(t, metrics, imp, per_vehicle, agree, conf, int(active.sum()), int(len(t)))


# ============================================================ saved models (product use)
import json
from dataclasses import field
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "database" / "models"
_MYT = timezone(timedelta(hours=8))


@dataclass
class ModelBundle:
    rf: object
    iso: object
    scaler: object
    train_raw: np.ndarray          # sorted anomaly raw scores of the training trips (for percentile ranks)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ModelView:
    """What the dashboard needs: a scored trip table and the version metadata."""
    table: pd.DataFrame
    meta: Dict[str, object]


def _merge_scored(feats: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    t = feats.merge(scored[["session_id", "score", "tier", "risk", "events"]], on="session_id", how="inner")
    return t[t["pings"] >= MIN_PINGS].reset_index(drop=True)


def fit_models(scored: pd.DataFrame, feats: pd.DataFrame, r0: float = 10.0, seed: int = 0,
               n_trees: int = 300) -> ModelBundle:
    """Fit the final Random Forest (risk-rate regressor) and Isolation Forest on all available trips."""
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    t = _merge_scored(feats, scored)
    active = (t["km"] >= MIN_ACTIVE_KM).to_numpy()
    if active.sum() < 20:
        raise ValueError(f"Need at least 20 driving trips (>= {MIN_ACTIVE_KM:.0f} km) to train; found {int(active.sum())}.")
    X = t[FEATURES].to_numpy(float)
    y = np.log1p(t["risk"].astype(float)).to_numpy()
    rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(X, y)
    scaler = RobustScaler().fit(np.log1p(np.abs(t.loc[active, IF_FEATURES].to_numpy(float))))
    Z = scaler.transform(np.log1p(np.abs(t.loc[active, IF_FEATURES].to_numpy(float))))
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(Z)
    raw = np.sort(-iso.score_samples(Z))
    return ModelBundle(rf, iso, scaler, raw, dict(n_trips=int(len(t)), n_driving=int(active.sum()), r0=r0, seed=seed))


def score_with(mb: ModelBundle, feats: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Score trips with a saved model: anomaly percentile (relative to its training trips) and a reason."""
    t = _merge_scored(feats, scored)
    active = (t["km"] >= MIN_ACTIVE_KM).to_numpy()
    t["anomaly"] = np.nan
    t["reason"] = ""
    if active.any():
        Z = mb.scaler.transform(np.log1p(np.abs(t.loc[active, IF_FEATURES].to_numpy(float))))
        raw = -mb.iso.score_samples(Z)
        t.loc[active, "anomaly"] = np.searchsorted(mb.train_raw, raw, side="right") / len(mb.train_raw) * 100.0
        t.loc[active, "reason"] = _reasons(Z)
    return t


# --------------------------------------------------------------------- registry
def _registry_file(model_dir) -> Path:
    return Path(model_dir) / "registry.json"


def _read_registry(model_dir) -> Dict[str, object]:
    f = _registry_file(model_dir)
    if not f.exists():
        return {"active": None, "versions": []}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"active": None, "versions": []}


def list_versions(model_dir=DEFAULT_MODEL_DIR) -> List[Dict[str, object]]:
    """Saved model versions, newest first."""
    return sorted(_read_registry(model_dir)["versions"], key=lambda m: m["version"], reverse=True)


def active_version(model_dir=DEFAULT_MODEL_DIR) -> Optional[Dict[str, object]]:
    reg = _read_registry(model_dir)
    return next((m for m in reg["versions"] if m["version"] == reg["active"]), None)


def set_active(version: int, model_dir=DEFAULT_MODEL_DIR) -> None:
    reg = _read_registry(model_dir)
    if not any(m["version"] == version for m in reg["versions"]):
        raise KeyError(f"Unknown model version {version}")
    reg["active"] = version
    _registry_file(model_dir).write_text(json.dumps(reg, indent=2), encoding="utf-8")


def load_model(version: int, model_dir=DEFAULT_MODEL_DIR) -> ModelBundle:
    import joblib
    meta = next((m for m in _read_registry(model_dir)["versions"] if m["version"] == version), None)
    if meta is None:
        raise KeyError(f"Unknown model version {version}")
    mb = joblib.load(Path(model_dir) / meta["file"])
    mb.meta = meta
    return mb


def train_and_save(scored: pd.DataFrame, feats: pd.DataFrame, r0: float = 10.0, date_range: str = "",
                   model_dir=DEFAULT_MODEL_DIR, seed: int = 0, n_trees: int = 300) -> Dict[str, object]:
    """Validate (leave-one-vehicle-out), fit the final models, save them as a new version and make it active."""
    import joblib
    res = run_ml(scored, feats, r0=r0, seed=seed, n_trees=n_trees)
    act = res.metrics[res.metrics["Subset"].str.startswith("Active")].set_index("Model")
    mb = fit_models(scored, feats, r0=r0, seed=seed, n_trees=n_trees)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    reg = _read_registry(model_dir)
    version = 1 + max([m["version"] for m in reg["versions"]], default=0)
    fname = f"risk_model_v{version}.joblib"
    meta = dict(
        mb.meta, version=version, file=fname, date_range=date_range,
        trained_at=datetime.now(_MYT).strftime("%Y-%m-%d %H:%M:%S"),
        n_vehicles=int(feats["vehicle_id"].nunique()),
        rf_within1=float(act.loc["Random Forest", "Tier_within1"]), rf_exact=float(act.loc["Random Forest", "Tier_exact"]),
        rf_spearman=float(act.loc["Random Forest", "Spearman"]), rf_mae=float(act.loc["Random Forest", "MAE_DSS"]),
        base_within1=float(act.loc["Median baseline", "Tier_within1"]),
        base_spearman=float(act.loc["Median baseline", "Spearman"]), base_mae=float(act.loc["Median baseline", "MAE_DSS"]),
        if_spearman=float(res.agreement.get("spearman", float("nan"))),
        if_top_overlap=float(res.agreement.get("top_overlap", float("nan"))))
    mb.meta = meta
    joblib.dump(mb, Path(model_dir) / fname)
    reg["versions"].append(meta)
    reg["active"] = version
    _registry_file(model_dir).write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return meta
