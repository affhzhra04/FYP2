"""Tests for the ML layer: feature extraction and model evaluation on synthetic fleets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dash_core as C  # noqa: E402
from src import ml_layer as M  # noqa: E402
from src import risk_engine as R  # noqa: E402


def synth_pings(n_vehicles=4, sessions_per=6, seed=1):
    """Aggressive sessions (fast, erratic) vs calm ones, so a good model must separate them."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vehicles):
        for d in range(1, sessions_per + 1):
            aggressive = d % 3 == 0
            n = 300
            ts = pd.Timestamp(f"2024-08-{d:02d} 09:00:00") + pd.to_timedelta(np.arange(n) * 30, unit="s")
            base = 105 if aggressive else 60
            speed = np.clip(base + rng.normal(0, 12 if aggressive else 4, n), 0, 130)
            lat = 3.0 + np.cumsum(speed) * 2e-4
            rows.append(pd.DataFrame(dict(vehicle_id=f"8_{d}_{v}", timestamp=ts, latitude=lat,
                                          longitude=101.6 + np.arange(n) * 1e-4, speed=speed,
                                          fuel_level=300.0 - np.arange(n) * 0.01, heading=rng.uniform(0, 360, n),
                                          ignition=1)))
    return pd.concat(rows, ignore_index=True)


def test_features_one_row_per_session_and_finite():
    df = R.prepare(synth_pings())
    f = M.session_features(df)
    assert len(f) == df["session_id"].nunique()
    assert set(M.FEATURES) <= set(f.columns)
    assert np.isfinite(f[M.FEATURES].to_numpy()).all()


def test_aggressive_sessions_have_higher_speed_features():
    f = M.session_features(R.prepare(synth_pings())).set_index("session_id")
    assert f.loc["8_3_0", "speed_p99"] > f.loc["8_1_0", "speed_p99"]
    assert f.loc["8_3_0", "fast_share"] > f.loc["8_1_0", "fast_share"]


def test_run_ml_leave_one_vehicle_out_beats_baseline():
    b = C.build_bundle(synth_pings())
    scored = C.session_scores(b, dict(R.DEFAULT_WEIGHTS))
    res = M.run_ml(scored, M.session_features(b.df), n_trees=60)
    act = res.metrics[res.metrics["Subset"].str.startswith("Active")].set_index("Model")
    assert act.loc["Random Forest", "MAE_DSS"] < act.loc["Median baseline", "MAE_DSS"]
    assert act.loc["Random Forest", "Spearman"] > 0.3
    assert not res.importance.empty and abs(res.importance["importance"].sum() - 1.0) < 1e-6
    assert res.table["ml_score"].between(0, 100).all()


def test_isolation_forest_scores_percentiles():
    b = C.build_bundle(synth_pings())
    res = M.run_ml(C.session_scores(b, dict(R.DEFAULT_WEIGHTS)), M.session_features(b.df), n_trees=40)
    a = res.table["anomaly"].dropna()
    assert len(a) > 0 and a.between(0, 100).all()


def test_model_registry_train_save_load_and_rollback():
    import tempfile
    d = Path(tempfile.mkdtemp())
    b = C.build_bundle(synth_pings())
    scored = C.session_scores(b, dict(R.DEFAULT_WEIGHTS))
    feats = M.session_features(b.df)
    assert M.active_version(d) is None and M.list_versions(d) == []
    m1 = M.train_and_save(scored, feats, date_range="2024-08", model_dir=d, n_trees=40)
    m2 = M.train_and_save(scored, feats, model_dir=d, n_trees=40)
    assert (m1["version"], m2["version"]) == (1, 2) and M.active_version(d)["version"] == 2
    assert [m["version"] for m in M.list_versions(d)] == [2, 1]
    M.set_active(1, d)
    assert M.active_version(d)["version"] == 1
    mb = M.load_model(1, d)
    tab = M.score_with(mb, feats, scored)
    a = tab["anomaly"].dropna()
    assert len(a) > 0 and a.between(0, 100).all() and (tab.loc[a.index, "reason"] != "").all()
    for key in ("rf_within1", "rf_spearman", "base_within1", "trained_at", "n_driving"):
        assert key in m1


def test_training_needs_enough_driving_trips():
    b = C.build_bundle(synth_pings(n_vehicles=1, sessions_per=3))
    scored = C.session_scores(b, dict(R.DEFAULT_WEIGHTS))
    try:
        M.fit_models(scored, M.session_features(b.df), n_trees=20)
    except ValueError as e:
        assert "driving trips" in str(e)
    else:
        raise AssertionError("expected ValueError")
